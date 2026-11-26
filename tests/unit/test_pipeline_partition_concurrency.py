"""run(): partitions run concurrently when gpu_jobs_per_device > 1 (sf4d-200 WS-2).

The output must be assembled exactly as in the sequential path, and with K > 1 the
partitions must actually overlap. Each stubbed partition waits for the other to
start, which can only succeed if they run at the same time.
"""

import threading
from types import SimpleNamespace

import pytest

from segflow4d.propagation.propagation_pipeline import PropagationPipeline


def _pipeline(jobs_per_device, run_partition):
    p = PropagationPipeline.__new__(PropagationPipeline)  # no RegistrationManager
    p._options = SimpleNamespace(gpu_jobs_per_device=jobs_per_device, write_result_to_disk=False)
    p._input = SimpleNamespace()
    p._validate_input = lambda: None
    partitions = [SimpleNamespace(name="sys", tps=[14, 15, 19]), SimpleNamespace(name="dias", tps=[8, 9, 15])]
    p._prepare_data = lambda: setattr(p, "_tp_partitions", partitions)
    p._run_partition = run_partition
    return p


def _result(partition):
    # tp 15 is in both partitions; the later partition must win, as in the loop
    return {tp: f"{partition.name}:{tp}" for tp in partition.tps}


def test_sequential_and_concurrent_assembly_are_identical():
    sequential = _pipeline(1, _result).run()
    concurrent = _pipeline(2, _result).run()
    assert concurrent == sequential
    assert list(concurrent) == list(sequential)
    assert sequential[15] == "dias:15"


def test_partitions_overlap_when_more_than_one_job_per_gpu():
    started = {"sys": threading.Event(), "dias": threading.Event()}

    def run_partition(partition):
        started[partition.name].set()
        other = "dias" if partition.name == "sys" else "sys"
        if not started[other].wait(timeout=5):
            raise AssertionError(f"{partition.name} ran without {other} running alongside")
        return _result(partition)

    assert _pipeline(2, run_partition).run()[19] == "sys:19"


def test_one_job_per_gpu_keeps_partitions_sequential():
    running = []

    def run_partition(partition):
        running.append(partition.name)
        assert len(running) == 1, "partitions overlapped with gpu_jobs_per_device=1"
        out = _result(partition)
        running.remove(partition.name)
        return out

    _pipeline(1, run_partition).run()
