"""
GPU worker functions for multiprocessing.

These functions must be at module level for pickling by ProcessPoolExecutor.
"""

import gc
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


def gpu_worker_init(device_id: int, registration_backend: str):
    """
    Initialize worker process with specific GPU.
    
    Args:
        device_id: GPU device to use.
        registration_backend: Registration backend name.
    """
    import torch
    import logging
    
    # Set up logging in worker process
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s - %(name)s - %(levelname)s] %(message)s'
    )
    
    # Set CUDA device for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    torch.cuda.set_device(0)  # Now device 0 is the only visible device
    
    logger = logging.getLogger(__name__)
    logger.info(f"Worker process initialized for GPU {device_id} (PID: {os.getpid()})")


def run_registration_job(job_data: Dict) -> Any:
    """
    Execute a registration job in a worker process.
    
    This function runs in a separate process with its own CUDA context.
    
    Args:
        job_data: Dictionary containing:
            - device_id: GPU device to use
            - method_name: Registration method to call
            - args: Positional arguments
            - kwargs: Keyword arguments
            - registration_backend: Backend name
            
    Returns:
        Result from the registration method.
    """
    import torch
    import gc
    import logging
    from registration.registration_handler import RegistrationHandlerFactory
    
    logger = logging.getLogger(__name__)
    
    device_id = job_data['device_id']
    method_name = job_data['method_name']
    args = job_data['args']
    kwargs = job_data['kwargs']
    registration_backend = job_data['registration_backend']
    
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    
    # Force CUDA to reinitialize in this process
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.set_device(0)  # Device 0 is now the mapped device
    
    logger.info(f"[PID {os.getpid()}] Running '{method_name}' on GPU {device_id}")
    
    try:
        # Create registration handler in this process
        handler = RegistrationHandlerFactory.create_registration_handler(registration_backend)
        
        # Get the method
        run_fn = getattr(handler, method_name, None)
        if run_fn is None:
            raise ValueError(f"Unknown method: {method_name}")
        
        # Execute
        result = run_fn(*args, **kwargs)
        
        logger.info(f"[PID {os.getpid()}] Completed '{method_name}' on GPU {device_id}")
        
        return result
        
    except Exception as e:
        logger.error(f"[PID {os.getpid()}] Error on GPU {device_id}: {e}", exc_info=True)
        raise
        
    finally:
        # Cleanup GPU memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()


def persistent_gpu_worker(device_id: int, registration_backend: str, 
                          job_queue, result_queue):
    """
    Persistent worker process for a specific GPU.
    
    Stays alive and processes jobs from the queue, avoiding process
    spawn overhead for each job.
    
    Args:
        device_id: GPU device to use.
        registration_backend: Registration backend name.
        job_queue: Multiprocessing queue for receiving jobs.
        result_queue: Multiprocessing queue for sending results.
    """
    import os
    import torch
    import logging
    from registration.registration_handler import RegistrationHandlerFactory
    
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    # Initialize GPU for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    torch.cuda.init()
    torch.cuda.set_device(0)
    
    # Create handler once (reused for all jobs)
    handler = RegistrationHandlerFactory.create_registration_handler(registration_backend)
    
    logger.info(f"[GPU {device_id}] Persistent worker started (PID: {os.getpid()})")
    
    while True:
        try:
            # Wait for job
            job_data = job_queue.get()
            
            if job_data is None:  # Shutdown signal
                logger.info(f"[GPU {device_id}] Worker shutting down")
                break
            
            job_id = job_data['job_id']
            method_name = job_data['method_name']
            args = job_data['args']
            kwargs = job_data['kwargs']
            
            logger.info(f"[GPU {device_id}] Processing job {job_id}: {method_name}")
            
            try:
                run_fn = getattr(handler, method_name)
                result = run_fn(*args, **kwargs)
                result_queue.put({'job_id': job_id, 'result': result, 'error': None})
            except Exception as e:
                logger.error(f"[GPU {device_id}] Job {job_id} failed: {e}")
                result_queue.put({'job_id': job_id, 'result': None, 'error': e})
            
            # Cleanup after each job
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"[GPU {device_id}] Worker error: {e}")
