import torch
import multiprocessing
import logging
from registration.registration_handler_factory import RegistrationHandlerFactory
from typing import Optional

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


class RegistrationManager:
    def __init__(self, registration_backend: str, number_of_concurrent_runners: Optional[int] = None):
        self._registration_handler = RegistrationHandlerFactory.create_registration_handler(registration_backend)
        self._device_type = self._registration_handler.get_device_type()
        
        # Auto-determine concurrent runners if not specified
        if number_of_concurrent_runners is None:
            if self._device_type == 'cuda':
                number_of_concurrent_runners = DeviceManager.get_gpu_count()
            else:
                number_of_concurrent_runners = DeviceManager.get_cpu_count()
        
        self._number_of_concurrent_runners = number_of_concurrent_runners
        logger.info(f"Device: {self._device_type}, Concurrent runners: {self._number_of_concurrent_runners}")



