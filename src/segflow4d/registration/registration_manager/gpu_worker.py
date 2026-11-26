"""
GPU worker functions for multiprocessing.

These functions must be at module level for pickling by ProcessPoolExecutor.
"""

import gc
import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Assumed CUDA context footprint when NVML cannot attribute memory to this process
# (e.g. a PID namespace hides it). Measured at about 300-450 MiB on an RTX 2080.
DEFAULT_CONTEXT_MB = 500.0


@dataclass
class JobOutcome:
    """What a worker returns: the handler's result plus the job's measured VRAM peak.

    peak_vram_mb is the allocator's peak reservation during the job plus this
    process's CUDA context, i.e. what the job needs on the device. The dispatcher
    uses it to decide how many jobs fit at once (gpu_jobs_per_device > 1).
    """
    result: Any
    peak_vram_mb: float | None = None


def _context_overhead_mb(torch) -> float:
    """GPU memory this process holds beyond what the torch allocator has reserved."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            if proc.pid == os.getpid() and proc.usedGpuMemory:
                return max(0.0, proc.usedGpuMemory / 2**20 - torch.cuda.memory_reserved() / 2**20)
    except Exception:  # NVML missing, or the process is not visible to it
        pass
    return DEFAULT_CONTEXT_MB


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


def run_registration_job(job_data: dict) -> Any:
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
    from segflow4d.registration.registration_handler import RegistrationHandlerFactory

    # Configure logging in this subprocess - not inherited from parent process
    log_level = job_data.get('log_level', logging.DEBUG)
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s - %(name)s - %(levelname)s] - %(message)s',
        force=True
    )

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
        
        # Execute, measuring the job's VRAM peak (see JobOutcome)
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        result = run_fn(*args, **kwargs)
        peak_vram_mb = None
        if cuda:
            peak_vram_mb = torch.cuda.max_memory_reserved() / 2**20 + _context_overhead_mb(torch)
        
        logger.info(f"[PID {os.getpid()}] Completed '{method_name}' on GPU {device_id}"
                    + (f" (peak {peak_vram_mb:.0f} MiB)" if peak_vram_mb is not None else ""))
        
        return JobOutcome(result=result, peak_vram_mb=peak_vram_mb)
        
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
    from segflow4d.registration.registration_handler import RegistrationHandlerFactory
    
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
