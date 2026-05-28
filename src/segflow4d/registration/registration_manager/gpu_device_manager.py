"""
GPU device management utilities.

Provides functionality for querying GPU information, memory usage,
and cleanup operations. Supports both CUDA (NVIDIA) and MPS (Apple Silicon).
"""

import gc
import logging
import multiprocessing

import torch

from segflow4d.utility import device_utils

try:
    import pynvml as pynvml_module
    PYNVML_AVAILABLE = True
except ImportError:
    pynvml_module = None
    PYNVML_AVAILABLE = False

logger = logging.getLogger(__name__)


class GPUDeviceManager:
    """
    Manages GPU device information and memory.

    On CUDA: uses NVML when available for accurate system-wide GPU memory
    reporting, falling back to PyTorch's memory tracking otherwise.

    On MPS: reports driver-allocated memory and the soft cap reported by
    ``recommendedMaxWorkingSetSize`` (which Torch exposes as the device
    total). Apple Silicon shares memory between CPU and GPU.
    """

    _nvml_initialized = False

    @classmethod
    def _init_nvml(cls):
        """Initialize NVIDIA Management Library."""
        if not cls._nvml_initialized and PYNVML_AVAILABLE and pynvml_module is not None:
            try:
                pynvml_module.nvmlInit()
                cls._nvml_initialized = True
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {e}")

    @staticmethod
    def get_cpu_count() -> int:
        """Get number of CPU cores available."""
        return multiprocessing.cpu_count()

    @staticmethod
    def get_gpu_count() -> int:
        """Get number of GPU devices available across CUDA or MPS."""
        return device_utils.device_count()

    @staticmethod
    def is_gpu_available() -> bool:
        """Check if any GPU accelerator (CUDA or MPS) is available."""
        return device_utils.is_accelerator_available()

    @staticmethod
    def get_gpu_memory_usage(device_id: int = 0) -> dict | None:
        """
        Get GPU memory usage for a specific device (in MB).

        Returns a dict with allocated_mb, reserved_mb, total_mb, free_mb,
        usage_percent, or None if no accelerator is available.
        """
        kind = device_utils.detect_device_kind()

        if kind == "cuda":
            if PYNVML_AVAILABLE and pynvml_module is not None:
                GPUDeviceManager._init_nvml()
                if GPUDeviceManager._nvml_initialized:
                    try:
                        handle = pynvml_module.nvmlDeviceGetHandleByIndex(device_id)
                        mem_info = pynvml_module.nvmlDeviceGetMemoryInfo(handle)
                        total_mb = mem_info.total / 1024 / 1024
                        used_mb = mem_info.used / 1024 / 1024
                        free_mb = mem_info.free / 1024 / 1024
                        return {
                            'allocated_mb': used_mb,
                            'reserved_mb': used_mb,
                            'total_mb': total_mb,
                            'free_mb': free_mb,
                            'usage_percent': (used_mb / total_mb) * 100
                        }
                    except Exception as e:
                        logger.warning(f"Failed to get NVML memory info: {e}")

            allocated = torch.cuda.memory_allocated(device_id) / 1024 / 1024
            reserved = torch.cuda.memory_reserved(device_id) / 1024 / 1024
            total = torch.cuda.get_device_properties(device_id).total_memory / 1024 / 1024
            free = total - allocated
            return {
                'allocated_mb': allocated,
                'reserved_mb': reserved,
                'total_mb': total,
                'free_mb': free,
                'usage_percent': (allocated / total) * 100
            }

        if kind == "mps":
            allocated = torch.mps.current_allocated_memory() / 1024 / 1024
            try:
                reserved = torch.mps.driver_allocated_memory() / 1024 / 1024
            except AttributeError:
                reserved = allocated
            try:
                total = torch.mps.recommended_max_memory() / 1024 / 1024
            except AttributeError:
                # Older torch builds — fall back to system RAM as a coarse upper bound.
                import psutil  # type: ignore
                total = psutil.virtual_memory().total / 1024 / 1024
            free = max(0.0, total - reserved)
            return {
                'allocated_mb': allocated,
                'reserved_mb': reserved,
                'total_mb': total,
                'free_mb': free,
                'usage_percent': (reserved / total) * 100 if total > 0 else 0.0,
            }

        return None

    @staticmethod
    def has_sufficient_vram(required_mb: float, device_id: int = 0,
                           safety_margin_mb: float = 2048) -> bool:
        """Check whether the device has enough free VRAM."""
        if not GPUDeviceManager.is_gpu_available():
            return False

        mem_info = GPUDeviceManager.get_gpu_memory_usage(device_id)
        if mem_info is None:
            return False

        safe_free = max(0, mem_info['free_mb'] - safety_margin_mb)
        return safe_free >= required_mb

    @staticmethod
    def cleanup_gpu_memory(device_id: int = 0) -> dict:
        """GPU memory cleanup for a specific device."""
        kind = device_utils.detect_device_kind()
        if kind == "cpu":
            return {}

        if kind == "cuda":
            current_device = torch.cuda.current_device()
            torch.cuda.set_device(device_id)
            try:
                for _ in range(5):
                    gc.collect()
                torch.cuda.synchronize(device_id)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device_id)
                torch.cuda.reset_accumulated_memory_stats(device_id)
                torch.cuda.synchronize(device_id)
                gc.collect()
                torch.cuda.empty_cache()
                mem_info = GPUDeviceManager.get_gpu_memory_usage(device_id)
                if mem_info:
                    logger.debug(f"GPU {device_id} cleaned - Free: {mem_info['free_mb']:.0f}MB, "
                               f"Usage: {mem_info['usage_percent']:.1f}%")
                    return mem_info
                return {}
            finally:
                torch.cuda.set_device(current_device)

        # MPS
        for _ in range(3):
            gc.collect()
        device_utils.synchronize(kind="mps")
        device_utils.empty_cache(kind="mps")
        gc.collect()
        mem_info = GPUDeviceManager.get_gpu_memory_usage(device_id)
        return mem_info or {}
