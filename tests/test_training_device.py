"""Tests for reporting which device trains the model.

A fake torch module drives every branch, so these run the same on machines with and without a GPU.
"""

import json
from types import SimpleNamespace

from http_attack_agent.training import describe_device


class _Device:
    def __init__(self, kind: str) -> None:
        self.type = kind

    def __str__(self) -> str:
        return self.type


def _torch(version: str, cuda: str | None, gpu: str = "NVIDIA Test GPU"):
    return SimpleNamespace(
        __version__=version,
        version=SimpleNamespace(cuda=cuda),
        cuda=SimpleNamespace(get_device_name=lambda device: gpu),
    )


def test_gpu_training_is_reported_without_a_warning():
    info, warning = describe_device(_torch("2.14.0+cu130", "13.0"), _Device("cuda"), False)

    assert warning is None
    assert info == {
        "device": "cuda",
        "device_name": "NVIDIA Test GPU",
        "torch": "2.14.0+cu130",
        "torch_cuda": "13.0",
    }


def test_requested_cpu_training_is_not_a_warning():
    info, warning = describe_device(_torch("2.14.0+cpu", None), _Device("cpu"), True)

    assert warning is None
    assert info["device"] == "cpu"
    assert info["device_name"] is None


def test_cpu_only_build_explains_the_missing_cuda_support():
    """The Windows default: pip installs a CPU wheel and training silently slows down."""

    _, warning = describe_device(_torch("2.14.0+cpu", None), _Device("cpu"), False)

    assert warning is not None
    assert warning.startswith("WARNING: training on CPU")
    assert "2.14.0+cpu" in warning
    assert "no CUDA support" in warning
    assert "GPU support" in warning, "the warning must point to the README fix"


def test_cuda_build_without_a_gpu_names_the_cuda_version():
    _, warning = describe_device(_torch("2.14.0+cu130", "13.0"), _Device("cpu"), False)

    assert warning is not None
    assert "CUDA 13.0" in warning
    assert "no usable GPU" in warning


def test_report_is_json_serializable_for_logs_and_metadata():
    info, _ = describe_device(_torch("2.14.0+cpu", None), _Device("cpu"), False)

    assert json.loads(json.dumps(info)) == info
