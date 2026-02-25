"""
GPU device management utilities.

Provides functionality for querying GPU information, memory usage,
and cleanup operations.
"""

import gc
import logging
import multiprocessing
from typing import Dict, Optional

import torch

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
    
    Uses NVML when available for accurate system-wide GPU memory reporting,
    falling back to PyTorch's memory tracking otherwise.
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
        """Get number of GPU devices available."""
        return torch.cuda.device_count()
    
    @staticmethod
    def is_gpu_available() -> bool:
        """Check if GPU is available."""
        return torch.cuda.is_available()
    
    @staticmethod
    def get_gpu_memory_usage(device_id: int = 0) -> Optional[Dict]:
        """
        Get GPU memory usage for a specific device (in MB).
        
        Uses NVML for system-wide memory tracking (includes all processes),
        falling back to PyTorch memory tracking if NVML is unavailable.
        
        Args:
            device_id: GPU device ID to query.
            
        Returns:
            Dictionary with memory info or None if unavailable:
            - allocated_mb: Memory currently allocated
            - reserved_mb: Memory reserved by PyTorch
            - total_mb: Total GPU memory
            - free_mb: Free memory available
            - usage_percent: Percentage of memory used
        """
        if not torch.cuda.is_available():
            return None
        
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
        
        # Fallback to PyTorch memory tracking
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
    
    @staticmethod
    def has_sufficient_vram(required_mb: float, device_id: int = 0, 
                           safety_margin_mb: float = 2048) -> bool:
        """
        Check if GPU has sufficient free VRAM.
        
        Args:
            required_mb: Required VRAM in MB.
            device_id: GPU device ID to check.
            safety_margin_mb: Additional safety margin in MB.
            
        Returns:
            True if sufficient VRAM is available.
        """
        if not torch.cuda.is_available():
            return False
        
        mem_info = GPUDeviceManager.get_gpu_memory_usage(device_id)
        if mem_info is None:
            return False
        
        safe_free = max(0, mem_info['free_mb'] - safety_margin_mb)
        return safe_free >= required_mb
    
    @staticmethod
    def cleanup_gpu_memory(device_id: int = 0) -> Dict:
        """
        GPU memory cleanup for a specific device.
        
        Performs garbage collection, cache clearing, and memory stats reset.
        
        Args:
            device_id: GPU device ID to clean up.
            
        Returns:
            Memory info after cleanup.
        """
        if not torch.cuda.is_available():
            return {}
        
        current_device = torch.cuda.current_device()
        torch.cuda.set_device(device_id)
        
        try:
            # Force Python garbage collection first (multiple rounds)
            for _ in range(5):
                gc.collect()
            
            # Synchronize to ensure all GPU operations complete
            torch.cuda.synchronize(device_id)
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats(device_id)
            torch.cuda.reset_accumulated_memory_stats(device_id)
            
            # Additional synchronization
            torch.cuda.synchronize(device_id)
            
            # One more gc.collect after CUDA ops
            gc.collect()
            
            # Try empty_cache again in case second gc.collect freed more
            torch.cuda.empty_cache()
            
            mem_info = GPUDeviceManager.get_gpu_memory_usage(device_id)
            if mem_info:
                logger.debug(f"GPU {device_id} cleaned - Free: {mem_info['free_mb']:.0f}MB, "
                           f"Usage: {mem_info['usage_percent']:.1f}%")
                return mem_info
            return {}
        finally:
            torch.cuda.set_device(current_device)
