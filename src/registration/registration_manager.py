import torch
import multiprocessing
import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future
from registration.registration_handler_factory import RegistrationHandlerFactory
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)


class DeviceManager:
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
        """Get GPU memory usage for a specific device (in MB)"""
        if not torch.cuda.is_available():
            return None
        
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
    def has_sufficient_vram(required_mb: float, device_id: int = 0, threshold_percent: float = 10.0) -> bool:
        """Check if GPU has sufficient free VRAM"""
        if not torch.cuda.is_available():
            return False
        
        mem_info = DeviceManager.get_gpu_memory_usage(device_id)
        if mem_info is None:
            return False
        
        safe_free = mem_info['free_mb'] * (1 - threshold_percent / 100)
        return safe_free >= required_mb


class RegistrationManager:
    def __init__(self, registration_backend: str, number_of_concurrent_runners: Optional[int] = None,
                 required_vram_mb: float = 10240, vram_check_interval: float = 5.0):
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
        
        logger.info(f"Device: {self._device_type}, Concurrent runners: {self._number_of_concurrent_runners}")
    
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
            # GPU: wrap with VRAM wait logic
            def run_with_vram_wait(*args, **kwargs):
                gpu_count = DeviceManager.get_gpu_count()
                device_id = None
                
                while device_id is None:
                    # Check all GPUs for sufficient VRAM
                    for gpu_id in range(gpu_count):
                        if DeviceManager.has_sufficient_vram(self._required_vram_mb, device_id=gpu_id):
                            device_id = gpu_id
                            logger.info(f"GPU {device_id} has sufficient VRAM, using this device")
                            break
                    
                    if device_id is None:
                        # Log status of all GPUs
                        for gpu_id in range(gpu_count):
                            mem_info = DeviceManager.get_gpu_memory_usage(gpu_id)
                            if mem_info is not None:
                                logger.warning(f"GPU {gpu_id} usage: {mem_info['usage_percent']:.1f}%. "
                                            f"Free: {mem_info['free_mb']:.0f}MB, Required: {self._required_vram_mb}MB")
                            else:
                                logger.warning(f"GPU {gpu_id} memory info not available")
                        logger.warning(f"No GPU has sufficient VRAM. Waiting {self._vram_check_interval}s...")
                        time.sleep(self._vram_check_interval)
                
                logger.info(f"Starting registration job on GPU {device_id}")
                # Set the device for PyTorch operations
                torch.cuda.set_device(device_id)
                return run_fn(*args, **kwargs)
            
            return self._executor.submit(run_with_vram_wait, *args, **kwargs)
        else:
            # CPU: submit directly without VRAM checks
            logger.info("Submitting registration job to CPU")
            return self._executor.submit(run_fn, *args, **kwargs)
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor"""
        self._executor.shutdown(wait=wait)



