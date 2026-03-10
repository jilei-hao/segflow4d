"""
CPU-based registration manager using ProcessPoolExecutor.

Each job runs in an isolated worker process so that CPU-bound registration
backends (e.g. Greedy) can execute in parallel without GIL contention, and
any accidental GPU initialisation in a backend cannot interfere with other
workers.
"""

import logging
import multiprocessing
from concurrent.futures import Future, ProcessPoolExecutor

from segflow4d.registration.registration_manager.abstract_registration_manager import AbstractRegistrationManager
from segflow4d.registration.registration_manager.cpu_worker import run_cpu_registration_job

logger = logging.getLogger(__name__)


class CPURegistrationManager(AbstractRegistrationManager):
    """
    CPU-based registration manager backed by :class:`ProcessPoolExecutor`.

    Each submitted job is serialised into a plain ``dict`` and dispatched to a
    worker process via :func:`~segflow4d.registration.registration_manager.cpu_worker.run_cpu_registration_job`.
    The worker creates its own handler instance, executes the method, and
    returns a picklable result.

    This design is consistent with :class:`GPURegistrationManager`'s
    ``ProcessJobDispatcher`` and makes it straightforward to add further
    CPU-only backends (ITK Elastix, ANTs, etc.) simply by implementing
    :class:`~segflow4d.registration.registration_handler.abstract_registration_handler.AbstractRegistrationHandler`
    and registering the backend in
    :class:`~segflow4d.registration.registration_handler.registration_handler_factory.RegistrationHandlerFactory`.
    """

    def __init__(self, registration_backend: str, max_workers: int | None = None):
        """
        Initialise the CPU registration manager.

        Args:
            registration_backend: Backend key (e.g. ``'greedy'``).
            max_workers: Maximum number of concurrent worker processes.
                Defaults to :func:`multiprocessing.cpu_count`.
        """
        self._registration_backend = registration_backend

        if max_workers is None:
            max_workers = multiprocessing.cpu_count()

        self._max_workers = max_workers
        self._executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )

        logger.info(
            f"CPURegistrationManager initialized - Workers: {max_workers}, "
            f"Backend: {registration_backend}"
        )

    def submit(self, method_name: str, *args, **kwargs) -> Future:
        """
        Submit a registration job to the process pool.

        Args:
            method_name: Name of the registration method to call on the handler.
            *args: Positional arguments forwarded to the method.
            **kwargs: Keyword arguments forwarded to the method.
                Any GPU-specific keys (e.g. ``required_vram_mb``) are silently
                removed so that CPU managers can be used as drop-in replacements.

        Returns:
            A :class:`~concurrent.futures.Future` that resolves to the
            registration result when the job completes.
        """
        # Drop GPU-only kwargs that have no meaning on CPU
        kwargs.pop('required_vram_mb', None)

        job_data = {
            'method_name': method_name,
            'args': args,
            'kwargs': kwargs,
            'registration_backend': self._registration_backend,
            'log_level': logging.getLogger().level,
        }

        logger.debug(f"Submitting CPU job: {method_name}")
        return self._executor.submit(run_cpu_registration_job, job_data)

    def get_queue_size(self) -> int:
        """
        Return current pending job count.

        Note: :class:`ProcessPoolExecutor` does not expose its internal queue,
        so this always returns 0.
        """
        return 0

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the process pool."""
        logger.info("CPURegistrationManager shutting down...")
        self._executor.shutdown(wait=wait)
        logger.info("CPURegistrationManager shutdown complete")
