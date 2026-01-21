"""
Abstract base class for registration managers.
"""

from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import Dict


class AbstractRegistrationManager(ABC):
    """Abstract base class for registration managers."""
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the manager and release resources.
        
        Args:
            wait: If True, wait for pending jobs to complete.
        """
        pass
    
    @abstractmethod
    def get_queue_size(self) -> int:
        """
        Get current number of pending jobs.
        
        Returns:
            Number of jobs in the queue.
        """
        pass
    
    def get_device_status(self) -> Dict[int, Dict]:
        """
        Get status of all devices (GPU only).
        
        Returns:
            Dictionary mapping device_id to status info.
        """
        return {}
