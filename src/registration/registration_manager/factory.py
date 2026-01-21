"""
Factory and singleton for creating registration managers.
"""

import logging
from typing import Dict, Optional

import torch
from concurrent.futures import Future

from registration.registration_manager.abstract_registration_manager import AbstractRegistrationManager
from registration.registration_manager.cpu_registration_manager import CPURegistrationManager
from registration.registration_manager.gpu_registration_manager import GPURegistrationManager

logger = logging.getLogger(__name__)


class RegistrationManagerFactory:
    """
    Factory for creating the appropriate registration manager.
    
    Automatically selects GPU or CPU manager based on hardware availability.
    """
    
    @staticmethod
    def create(registration_backend: str,
               max_workers: Optional[int] = None,
               required_vram_mb: float = 10240,
               vram_check_interval: float = 0.1,
               use_processes: bool = True,
               use_persistent_workers: bool = False,
               force_cpu: bool = False) -> AbstractRegistrationManager:
        """
        Create a registration manager based on available hardware.
        
        Args:
            registration_backend: Backend to use ('fireants', 'greedy', etc.)
            max_workers: Maximum concurrent workers.
            required_vram_mb: Required VRAM per job (GPU only).
            vram_check_interval: VRAM check interval in seconds (GPU only).
            use_processes: Use process-based GPU execution (recommended).
            use_persistent_workers: Use persistent worker processes (GPU only).
            force_cpu: Force CPU even if GPU is available.
            
        Returns:
            Appropriate registration manager instance.
        """
        if not force_cpu and torch.cuda.is_available():
            logger.info("Creating GPU registration manager")
            return GPURegistrationManager(
                registration_backend=registration_backend,
                max_workers=max_workers,
                required_vram_mb=required_vram_mb,
                vram_check_interval=vram_check_interval,
                use_processes=use_processes,
                use_persistent_workers=use_persistent_workers
            )
        else:
            logger.info("Creating CPU registration manager")
            return CPURegistrationManager(
                registration_backend=registration_backend,
                max_workers=max_workers
            )


class RegistrationManager:
    """
    Singleton wrapper for registration manager.
    
    Provides a global access point for registration job submission.
    Automatically creates the appropriate manager type based on hardware.
    """
    
    _instance: Optional['RegistrationManager'] = None
    _initialized: bool = False
    
    def __new__(cls, registration_backend: Optional[str] = None,
                number_of_concurrent_runners: Optional[int] = None,
                required_vram_mb: float = 10240,
                vram_check_interval: float = 0.1,
                use_processes: bool = True,
                use_persistent_workers: bool = False,
                force_cpu: bool = False):
        if cls._instance is None:
            cls._instance = super(RegistrationManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, registration_backend: Optional[str] = None,
                 number_of_concurrent_runners: Optional[int] = None,
                 required_vram_mb: float = 10240,
                 vram_check_interval: float = 0.1,
                 use_processes: bool = True,
                 use_persistent_workers: bool = False,
                 force_cpu: bool = False):
        """
        Initialize the registration manager singleton.
        
        Args:
            registration_backend: Backend to use ('fireants', 'greedy', etc.)
            number_of_concurrent_runners: Maximum concurrent workers.
            required_vram_mb: Required VRAM per job (GPU only).
            vram_check_interval: VRAM check interval in seconds.
            use_processes: Use process-based GPU execution (recommended).
            use_persistent_workers: Use persistent worker processes.
            force_cpu: Force CPU even if GPU is available.
        """
        if RegistrationManager._initialized:
            return
        
        if registration_backend is None:
            raise ValueError("registration_backend is required for first initialization")
        
        self._manager = RegistrationManagerFactory.create(
            registration_backend=registration_backend,
            max_workers=number_of_concurrent_runners,
            required_vram_mb=required_vram_mb,
            vram_check_interval=vram_check_interval,
            use_processes=use_processes,
            use_persistent_workers=use_persistent_workers,
            force_cpu=force_cpu
        )
        
        RegistrationManager._initialized = True
        logger.info("RegistrationManager singleton initialized")
    
    @staticmethod
    def get_instance() -> 'RegistrationManager':
        """
        Get the singleton instance.
        
        Raises:
            ValueError: If not initialized yet.
        """
        if RegistrationManager._instance is None:
            raise ValueError("RegistrationManager is not initialized yet")
        return RegistrationManager._instance
    
    def submit(self, method_name: str, *args, **kwargs) -> Future:
        """
        Submit a registration job.
        
        Args:
            method_name: Name of the registration method to call.
            *args: Positional arguments for the method.
            **kwargs: Keyword arguments for the method.
            
        Returns:
            Future that will contain the result when the job completes.
        """
        return self._manager.submit(method_name, *args, **kwargs)
    
    def get_queue_size(self) -> int:
        """Get current number of pending jobs."""
        return self._manager.get_queue_size()
    
    def get_device_status(self) -> Dict[int, Dict]:
        """Get status of all GPU devices."""
        return self._manager.get_device_status()
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown all workers."""
        self._manager.shutdown(wait=wait)
        logger.info("RegistrationManager shutdown complete")
    
    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton.
        
        Useful for testing or reinitializing with different settings.
        """
        if cls._instance is not None:
            cls._instance.shutdown(wait=True)
        cls._instance = None
        cls._initialized = False
        logger.info("RegistrationManager reset")
