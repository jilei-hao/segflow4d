"""
CPU-based registration manager using ThreadPoolExecutor.
"""

import logging
import multiprocessing
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

from registration.registration_manager.abstract_registration_manager import AbstractRegistrationManager

logger = logging.getLogger(__name__)


class CPURegistrationManager(AbstractRegistrationManager):
    """
    Simple CPU-based registration manager using ThreadPoolExecutor.
    
    This manager is used when CUDA is not available or when CPU execution
    is explicitly requested.
    """
    
    def __init__(self, registration_backend: str, max_workers: Optional[int] = None):
        """
        Initialize CPU registration manager.
        
        Args:
            registration_backend: Backend to use ('fireants', 'greedy', etc.)
            max_workers: Maximum number of concurrent workers.
                        Defaults to CPU count.
        """
        from registration.registration_handler.registration_handler_factory import RegistrationHandlerFactory
        
        self._registration_backend = registration_backend
        self._registration_handler = RegistrationHandlerFactory.create_registration_handler(registration_backend)
        
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        
        self._max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="CPURegistration"
        )
        
        logger.info(f"CPURegistrationManager initialized - Workers: {max_workers}, "
                   f"Backend: {registration_backend}")
    
    def submit(self, method_name: str, *args, **kwargs) -> Future:
        """
        Submit a registration job to the thread pool.
        
        Args:
            method_name: Name of the registration method to call.
            *args: Positional arguments for the method.
            **kwargs: Keyword arguments for the method.
            
        Returns:
            Future that will contain the result when the job completes.
        """
        # Remove GPU-specific kwargs if present
        kwargs.pop('required_vram_mb', None)
        
        run_fn = getattr(self._registration_handler, method_name, None)
        if run_fn is None:
            raise ValueError(f"Unknown method: {method_name}")
        
        logger.debug(f"Submitting CPU job: {method_name}")
        return self._executor.submit(run_fn, *args, **kwargs)
    
    def get_queue_size(self) -> int:
        """
        Get current queue size.
        
        Note: ThreadPoolExecutor doesn't expose queue size easily.
        """
        return 0
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool."""
        logger.info("CPURegistrationManager shutting down...")
        self._executor.shutdown(wait=wait)
        logger.info("CPURegistrationManager shutdown complete")
