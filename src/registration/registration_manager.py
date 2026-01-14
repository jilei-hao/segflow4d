import threading
import torch
import multiprocessing
import logging
import time
import gc
from concurrent.futures import ThreadPoolExecutor, Future
from registration.registration_handler_factory import RegistrationHandlerFactory
from typing import Optional, Callable, Any

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

logger = logging.getLogger(__name__)


class DeviceManager:
    _nvml_initialized = False
    
    @staticmethod
    def _init_nvml():
        """Initialize NVIDIA Management Library"""
        if not DeviceManager._nvml_initialized and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                DeviceManager._nvml_initialized = True
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {e}")
    
    @staticmethod
    def get_cpu_count() -> int:
        """Get number of CPU cores available"""
        return multiprocessing.cpu_count()
    
    @staticmethod
    def get_gpu_count() -> int:
        """Get number of GPU devices available"""
        return torch.cuda.device_count()
    
    @staticmethod
    def is_gpu_available() -> bool:
        """Check if GPU is available"""
        return torch.cuda.is_available()
    
    @staticmethod
    def get_device_info() -> dict:
        """Get detailed device information"""
        info = {
            'cpu_count': multiprocessing.cpu_count(),
            'gpu_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        return info
    
    @staticmethod
    def get_gpu_memory_usage(device_id: int = 0) -> Optional[dict]:
        """Get GPU memory usage for a specific device (in MB) - includes all processes"""
        if not torch.cuda.is_available():
            return None
        
        # Try to get real GPU memory usage from NVML first
        if PYNVML_AVAILABLE:
            DeviceManager._init_nvml()
            if DeviceManager._nvml_initialized:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                    
                    # Force garbage collection on the target device to free memory
                    current_device = torch.cuda.current_device()
                    torch.cuda.set_device(device_id)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.set_device(current_device)
                    
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_mb = mem_info.total / 1024 / 1024
                    used_mb = mem_info.used / 1024 / 1024
                    free_mb = mem_info.free / 1024 / 1024
                    
                    return {
                        'allocated_mb': used_mb,  # Total used by all processes
                        'reserved_mb': used_mb,   # Same as allocated for NVML
                        'total_mb': total_mb,
                        'free_mb': free_mb,
                        'usage_percent': (used_mb / total_mb) * 100
                    }
                except Exception as e:
                    logger.warning(f"Failed to get NVML memory info: {e}")
        
        # Fallback to PyTorch memory tracking (less accurate)
        allocated = torch.cuda.memory_allocated(device_id) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(device_id) / 1024 / 1024
        total = torch.cuda.get_device_properties(device_id).total_memory / 1024 / 1024
        free = total - allocated
        
        logger.warning("Using PyTorch memory tracking - may not reflect other processes")
        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'total_mb': total,
            'free_mb': free,
            'usage_percent': (allocated / total) * 100
        }
    
    @staticmethod
    def has_sufficient_vram(required_mb: float, device_id: int = 0, threshold_percent: float = 10.0) -> bool:
        """Check if GPU has sufficient free VRAM"""
        if not torch.cuda.is_available():
            return False
        
        mem_info = DeviceManager.get_gpu_memory_usage(device_id)
        if mem_info is None:
            return False
        
        logger.debug(f"GPU {device_id} Memory Info: {mem_info}")
        
        # Use free memory minus safety margin for fragmentation, etc.
        safety_margin_mb = 2048  # Reduced from 3072 to 2GB - your GPU 0 has 22GB free
        safe_free = max(0, mem_info['free_mb'] - safety_margin_mb)
        
        logger.info(f"GPU {device_id} safe free memory: {safe_free:.0f}MB, required: {required_mb}MB")
        return safe_free >= required_mb
    
    @staticmethod
    def cleanup_gpu_memory(device_id: int = 0):
        """Aggressive GPU memory cleanup for a specific device"""
        if not torch.cuda.is_available():
            return
        
        current_device = torch.cuda.current_device()
        torch.cuda.set_device(device_id)
        
        try:
            # Python garbage collection first
            gc.collect()
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Synchronize to ensure all operations complete
            torch.cuda.synchronize()
            
            # Log memory status
            mem_info = DeviceManager.get_gpu_memory_usage(device_id)
            if mem_info:
                logger.debug(f"GPU {device_id} cleaned - Allocated: {mem_info['allocated_mb']:.0f}MB, "
                           f"Free: {mem_info['free_mb']:.0f}MB, Usage: {mem_info['usage_percent']:.1f}%")
        finally:
            # Restore original device
            torch.cuda.set_device(current_device)


class RegistrationManager:
    _instance = None
    _initialized = False
    
    def __new__(cls, registration_backend: Optional[str] = None, number_of_concurrent_runners: Optional[int] = None,
                required_vram_mb: float = 10240, vram_check_interval: float = 5.0):
        if cls._instance is None:
            cls._instance = super(RegistrationManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, registration_backend: Optional[str] = None, number_of_concurrent_runners: Optional[int] = None,
                 required_vram_mb: float = 10240, vram_check_interval: float = 5.0):
        # Only initialize once
        if RegistrationManager._initialized:
            return
        
        if registration_backend is None:
            raise ValueError("registration_backend is required for first initialization")
        
        self._registration_handler = RegistrationHandlerFactory.create_registration_handler(registration_backend)
        self._device_type = self._registration_handler.get_device_type()
        
        # Auto-determine concurrent runners if not specified
        if number_of_concurrent_runners is None:
            if self._device_type == 'cuda':
                number_of_concurrent_runners = DeviceManager.get_gpu_count()
            else:
                number_of_concurrent_runners = DeviceManager.get_cpu_count()
        
        self._number_of_concurrent_runners = number_of_concurrent_runners
        self._required_vram_mb = required_vram_mb
        self._vram_check_interval = vram_check_interval
        self._executor = ThreadPoolExecutor(max_workers=self._number_of_concurrent_runners)
        
        RegistrationManager._initialized = True
        logger.info(f"Device: {self._device_type}, Concurrent runners: {self._number_of_concurrent_runners}, Required VRAM: {self._required_vram_mb}MB")
    
    @staticmethod
    def get_instance():
        """Get the singleton instance of RegistrationManager"""
        if RegistrationManager._instance is None:
            raise ValueError("RegistrationManager is not initialized yet")
        return RegistrationManager._instance

    def _get_registration_function(self, method_name: str) -> Callable[..., Any]:
        """
        Get the registration function to use.
        
        Args:
            method_name: Name of the registration method to use.
        
        Returns:
            The callable registration function.
            
        Raises:
            ValueError: If the specified method doesn't exist on the handler.
        """
        run_fn = getattr(self._registration_handler, method_name, None)
        if run_fn is None:
            raise ValueError(f"Registration handler does not have method '{method_name}'")
        logger.info(f"Using registration method: {method_name}")
        return run_fn
    
    def submit(self, method_name: str, *args, **kwargs) -> Future:
        """
        Submit a registration job.
        
        Args:
            method_name: Name of the registration method to use (required)
            *args: Positional arguments for the registration function
            **kwargs: Keyword arguments for the registration function
        
        Returns:
            A Future representing the submitted job.
        """
        run_fn = self._get_registration_function(method_name)
        
        if self._device_type == 'cuda':
            # GPU: wrap with VRAM wait logic and aggressive cleanup
            def run_with_vram_wait_and_cleanup(*args, **kwargs):
                gpu_count = DeviceManager.get_gpu_count()
                device_id = None
                thread_id = threading.get_ident()
                
                # Pre-cleanup: aggressive cleanup before looking for VRAM
                logger.debug(f"[Thread {thread_id}] Pre-registration cleanup...")
                for gpu_id in range(gpu_count):
                    DeviceManager.cleanup_gpu_memory(gpu_id)
                
                while device_id is None:
                    # Check all GPUs for sufficient VRAM
                    for gpu_id in range(gpu_count):
                        if DeviceManager.has_sufficient_vram(self._required_vram_mb, device_id=gpu_id):
                            device_id = gpu_id
                            logger.info(f"[Thread {thread_id}] GPU {device_id} has sufficient VRAM, required {self._required_vram_mb}MB")
                            break
                    
                    if device_id is None:
                        # Log status of all GPUs
                        for gpu_id in range(gpu_count):
                            mem_info = DeviceManager.get_gpu_memory_usage(gpu_id)
                            if mem_info is not None:
                                logger.warning(f"[Thread {thread_id}] GPU {gpu_id} usage: {mem_info['usage_percent']:.1f}%. "
                                            f"Free: {mem_info['free_mb']:.0f}MB, Required: {self._required_vram_mb}MB")
                            else:
                                logger.warning(f"[Thread {thread_id}] GPU {gpu_id} memory info not available")
                        
                        logger.warning(f"[Thread {thread_id}] No GPU has sufficient VRAM. Cleaning up and waiting {self._vram_check_interval}s...")
                        
                        # Aggressive cleanup before waiting
                        for gpu_id in range(gpu_count):
                            DeviceManager.cleanup_gpu_memory(gpu_id)
                        
                        time.sleep(self._vram_check_interval)
                
                logger.info(f"[Thread {thread_id}] Starting registration job on GPU {device_id}")
                
                # Set the device for PyTorch operations
                torch.cuda.set_device(device_id)
                
                try:
                    # Run the registration
                    result = run_fn(*args, **kwargs)
                    return result
                finally:
                    # Post-cleanup: aggressive cleanup after registration completes
                    logger.debug(f"[Thread {thread_id}] Post-registration cleanup on GPU {device_id}...")
                    DeviceManager.cleanup_gpu_memory(device_id)
            
            return self._executor.submit(run_with_vram_wait_and_cleanup, *args, **kwargs)
        else:
            # CPU: submit directly without VRAM checks
            logger.info("Submitting registration job to CPU")
            return self._executor.submit(run_fn, *args, **kwargs)
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor"""
        self._executor.shutdown(wait=wait)
    
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful for testing)"""
        if cls._instance is not None:
            cls._instance.shutdown(wait=True)
        cls._instance = None
        cls._initialized = False



