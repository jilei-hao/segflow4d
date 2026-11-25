"""Unit tests for cyclic_time chain building (sf4d-200 KNOWN_ISSUES B3).

Without ``cyclic_time``, ``_run_partition`` splits a group's timepoints by
numeric order. A group whose membership straddles the end of the series (e.g.
ref 8 owning ``..., 19, 20, 1, 2``) then reaches 1 and 2 by one large backwards
jump across frames owned by another group. On bav17 that jump (32 -> 3) left
TPs 1-3 at all-foreground Dice 0.46; with ``cyclic_time`` they reached 0.75-0.78.

``_build_cyclic_chains`` walks outward from the reference one frame at a time
and wraps around the series end, so every chain step is between adjacent frames.
No registration is run here; these tests pin down the chain topology only.
"""

import logging
from unittest.mock import MagicMock

import pytest

from segflow4d.common.types.tp_data import TPData
from segflow4d.propagation.propagation_pipeline import PropagationPipeline

PIPELINE_LOGGER = "segflow4d.propagation.propagation_pipeline"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pipeline(n_timepoints: int, cyclic_time: bool = True) -> PropagationPipeline:
    """A pipeline with only the state the chain logic reads.

    ``__init__`` is skipped on purpose: it creates the RegistrationManager
    singleton (and on a GPU box, a worker pool), none of which the chain
    logic touches.
    """
    pipeline = PropagationPipeline.__new__(PropagationPipeline)
    pipeline._n_timepoints = n_timepoints
    pipeline._options = MagicMock(cyclic_time=cyclic_time)
    return pipeline


def _is_adjacent(a: int, b: int, n: int) -> bool:
    return (b - a) % n in (1, n - 1)


def _assert_valid_chains(forward, backward, members, tp_ref, n):
    assert forward[0] == tp_ref and backward[0] == tp_ref
    for chain in (forward, backward):
        for a, b in zip(chain, chain[1:]):
            assert _is_adjacent(a, b, n), f"non-adjacent step {a} -> {b} in {chain}"
    assert set(forward) | set(backward) == set(members), "a member was not reached"
    assert not (set(forward[1:]) & set(backward[1:])), "a timepoint is propagated twice"


def _fake_partition(tp_ref: int, tp_targets: list[int]) -> MagicMock:
    partition = MagicMock()
    partition._input.tp_ref = tp_ref
    partition._input.tp_target = list(tp_targets)
    partition._tp_data = {tp: TPData() for tp in set(tp_targets) | {tp_ref}}
    return partition


def _run_partition_recording_chains(pipeline, partition) -> list[list[int]]:
    """Run _run_partition with the per-direction propagation stubbed out;
    return the tp_list each direction was asked to propagate."""
    chains: list[list[int]] = []

    def fake_unidirectional(tp_data, ref_input, tp_list, options):
        chains.append(list(tp_list))
        return {tp: TPData() for tp in tp_list}

    pipeline._run_unidirectional_propagation = fake_unidirectional
    pipeline._run_partition(partition)
    return chains


# ---------------------------------------------------------------------------
# _build_cyclic_chains
# ---------------------------------------------------------------------------

class TestBuildCyclicChains:
    def test_group_straddling_the_wrap_continues_through_the_series_end(self):
        n = 20
        members = sorted(list(range(8, 21)) + [1, 2])
        forward, backward = _pipeline(n)._build_cyclic_chains(members, tp_ref=8)

        assert forward == list(range(8, 21)) + [1, 2]
        assert backward == [8]
        _assert_valid_chains(forward, backward, members, 8, n)

    def test_reference_at_series_start_walks_backwards_through_the_end(self):
        n = 20
        members = [1, 2, 3, 18, 19, 20]
        forward, backward = _pipeline(n)._build_cyclic_chains(members, tp_ref=1)

        assert forward == [1, 2, 3]
        assert backward == [1, 20, 19, 18]
        _assert_valid_chains(forward, backward, members, 1, n)

    def test_group_away_from_the_wrap_matches_the_linear_split(self):
        forward, backward = _pipeline(20)._build_cyclic_chains([3, 4, 5, 6, 7, 8], tp_ref=5)

        assert forward == [5, 6, 7, 8]
        assert backward == [5, 4, 3]

    def test_full_loop_reaches_each_timepoint_once_by_its_shorter_route(self):
        n = 10
        members = list(range(1, n + 1))
        forward, backward = _pipeline(n)._build_cyclic_chains(members, tp_ref=3)

        # 3 -> 8 is 5 steps either way; the tie goes to the forward chain.
        assert forward == [3, 4, 5, 6, 7, 8]
        assert backward == [3, 2, 1, 10, 9]
        _assert_valid_chains(forward, backward, members, 3, n)

    def test_reference_only_group_yields_trivial_chains(self):
        forward, backward = _pipeline(20)._build_cyclic_chains([8], tp_ref=8)

        assert forward == [8]
        assert backward == [8]

    def test_unreachable_member_raises_instead_of_jumping(self):
        # 15 is separated from ref 8 by frames 11-14, which the group does not own.
        with pytest.raises(ValueError, match="cannot be reached"):
            _pipeline(20)._build_cyclic_chains([8, 9, 10, 15], tp_ref=8)


# ---------------------------------------------------------------------------
# _warn_on_chain_gaps (the non-cyclic path)
# ---------------------------------------------------------------------------

class TestChainGapWarning:
    def test_jump_across_foreign_frames_is_logged(self, caplog):
        with caplog.at_level(logging.WARNING, logger=PIPELINE_LOGGER):
            PropagationPipeline._warn_on_chain_gaps("backward", [8, 7, 2, 1])

        assert "7 -> 2 skips 4" in caplog.text

    def test_adjacent_chain_logs_nothing(self, caplog):
        with caplog.at_level(logging.WARNING, logger=PIPELINE_LOGGER):
            PropagationPipeline._warn_on_chain_gaps("forward", [3, 4, 5, 6])

        assert caplog.records == []


# ---------------------------------------------------------------------------
# _run_partition: the option actually selects the chain builder
# ---------------------------------------------------------------------------

class TestRunPartitionChains:
    TARGETS = list(range(9, 21)) + [1, 2]  # ref 8 owns 8..20 and wraps to 1, 2

    def test_cyclic_time_propagates_wrapped_frames_forward_through_the_end(self):
        chains = _run_partition_recording_chains(
            _pipeline(20, cyclic_time=True), _fake_partition(8, self.TARGETS)
        )

        # The backward chain is just [8], so only one direction runs.
        assert chains == [list(range(8, 21)) + [1, 2]]

    def test_without_cyclic_time_wrapped_frames_are_reached_by_a_jump(self, caplog):
        with caplog.at_level(logging.WARNING, logger=PIPELINE_LOGGER):
            chains = _run_partition_recording_chains(
                _pipeline(20, cyclic_time=False), _fake_partition(8, self.TARGETS)
            )

        # The pre-cyclic behaviour, kept as the default: numeric split, and a
        # backward chain whose 8 -> 2 step skips five frames this group does not own.
        assert sorted(chains) == sorted([list(range(8, 21)), [8, 2, 1]])
        assert "8 -> 2 skips 5" in caplog.text
