"""main(): both entry paths, YAML ``--config`` and plain CLI flags, must hand the
propagation options to ``PropagationInputFactory.set_options``.

sf4d-200 KNOWN_ISSUES B6: ``mask_high_res_registration`` could be set from YAML
but had no CLI flag, so a CLI run silently used the default. The factory and the
pipeline are mocked out here; nothing is read from disk and nothing is registered.
"""

import sys
from unittest.mock import MagicMock, patch

import yaml

import segflow4d.main as sf_main

CLI_REQUIRED = [
    "--image4d", "img.nii.gz",
    "--output", "out",
    "--tp-ref", "1",
    "--tp-targets", "2", "3",
    "--seg-ref", "seg.nii.gz",
]


def _set_options_kwargs(monkeypatch, argv: list[str]) -> dict:
    """Run main() with ``argv`` and return the kwargs it passed to set_options."""
    monkeypatch.setattr(sys, "argv", ["segflow4d", *argv])
    factory = MagicMock()
    with patch.object(sf_main, "PropagationInputFactory", return_value=factory), \
         patch.object(sf_main, "PropagationPipeline"), \
         patch.object(sf_main, "_install_signal_handlers"), \
         patch.object(sf_main, "async_writer"), \
         patch.object(sf_main, "configure_logging"):
        sf_main.main()
    factory.set_options.assert_called_once()
    return factory.set_options.call_args.kwargs


def _write_config(tmp_path, **options) -> str:
    config = {
        "image4d": "img.nii.gz",
        "output": str(tmp_path / "out"),
        "tp_input_groups": [{"tp_ref": 1, "tp_targets": [2, 3], "seg_ref": "seg.nii.gz"}],
        **options,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


class TestCliPath:
    def test_defaults(self, monkeypatch):
        kwargs = _set_options_kwargs(monkeypatch, CLI_REQUIRED)

        assert kwargs["cyclic_time"] is False
        assert kwargs["mask_high_res_registration"] is True
        assert kwargs["propagation_strategy_combo"] == "sequential_star"

    def test_cyclic_time_flag(self, monkeypatch):
        kwargs = _set_options_kwargs(monkeypatch, CLI_REQUIRED + ["--cyclic-time"])

        assert kwargs["cyclic_time"] is True

    def test_no_mask_high_res_registration_flag(self, monkeypatch):
        kwargs = _set_options_kwargs(
            monkeypatch, CLI_REQUIRED + ["--no-mask-high-res-registration"]
        )

        assert kwargs["mask_high_res_registration"] is False

    def test_sequential_sasd_combo_is_accepted(self, monkeypatch):
        kwargs = _set_options_kwargs(
            monkeypatch, CLI_REQUIRED + ["--propagation-strategy-combo", "sequential_sasd"]
        )

        assert kwargs["propagation_strategy_combo"] == "sequential_sasd"


class TestYamlPath:
    def test_defaults(self, monkeypatch, tmp_path):
        kwargs = _set_options_kwargs(monkeypatch, ["--config", _write_config(tmp_path)])

        assert kwargs["cyclic_time"] is False
        assert kwargs["mask_high_res_registration"] is True
        assert kwargs["propagation_strategy_combo"] == "sequential_star"

    def test_options_are_read(self, monkeypatch, tmp_path):
        config = _write_config(
            tmp_path,
            cyclic_time=True,
            mask_high_res_registration=False,
            propagation_strategy_combo="sequential_sasd",
        )
        kwargs = _set_options_kwargs(monkeypatch, ["--config", config])

        assert kwargs["cyclic_time"] is True
        assert kwargs["mask_high_res_registration"] is False
        assert kwargs["propagation_strategy_combo"] == "sequential_sasd"
