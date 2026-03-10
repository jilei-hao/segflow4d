"""
CPU worker functions for multiprocessing.

These functions must be at module level for pickling by ProcessPoolExecutor.
They mirror the interface of gpu_worker.py but do not perform any GPU/CUDA setup.
"""

import gc
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def run_cpu_registration_job(job_data: dict) -> Any:
    """
    Execute a registration job in a CPU worker process.

    Runs in a separate process, avoiding Python GIL contention for CPU-bound
    registration backends (e.g. Greedy). Each call creates a fresh handler
    instance so that there are no shared-state issues across workers.

    Args:
        job_data: Dictionary containing:
            - method_name (str): Name of the registration method to call.
            - args (tuple): Positional arguments passed to the method.
            - kwargs (dict): Keyword arguments passed to the method.
            - registration_backend (str): Backend name (e.g. 'greedy').
            - log_level (int): Python logging level for this subprocess.

    Returns:
        The return value of the registration method (must be picklable).

    Raises:
        ValueError: If the requested method does not exist on the handler.
        Exception: Any exception raised by the underlying registration handler.
    """
    from segflow4d.registration.registration_handler.registration_handler_factory import RegistrationHandlerFactory

    # Configure logging in this subprocess (not inherited from parent)
    log_level = job_data.get('log_level', logging.DEBUG)
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s - %(name)s - %(levelname)s] - %(message)s',
        force=True
    )

    local_logger = logging.getLogger(__name__)

    method_name = job_data['method_name']
    args = job_data['args']
    kwargs = job_data['kwargs']
    registration_backend = job_data['registration_backend']

    local_logger.info(f"[PID {os.getpid()}] Running '{method_name}' via CPU backend '{registration_backend}'")

    try:
        handler = RegistrationHandlerFactory.create_registration_handler(registration_backend)

        run_fn = getattr(handler, method_name, None)
        if run_fn is None:
            raise ValueError(f"Unknown method '{method_name}' on handler for backend '{registration_backend}'")

        result = run_fn(*args, **kwargs)

        local_logger.info(f"[PID {os.getpid()}] Completed '{method_name}'")
        return result

    except Exception as e:
        local_logger.error(f"[PID {os.getpid()}] Error in '{method_name}': {e}", exc_info=True)
        raise

    finally:
        gc.collect()
