import threading
import torch
import multiprocessing
import logging
import time
import gc
import os
import sys
import signal
from queue import Queue, Empty
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Optional, Callable, Any, Dict, List, Tuple
import pickle

try:
    import pynvml as pynvml_module
    PYNVML_AVAILABLE = True
except ImportError:
    pynvml_module = None
    PYNVML_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Worker function for ProcessPoolExecutor (must be at module level)
# ============================================================================

def _gpu_worker_init(device_id: int, registration_backend: str):
    """Initialize worker process with specific GPU"""
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


def _run_registration_job(job_data: Dict) -> Any:
    """
    Execute a registration job in a worker process.
    
    This function runs in a separate process with its own CUDA context.
    """
    import torch
    import gc
    import logging
    from registration.registration_handler_factory import RegistrationHandlerFactory
    
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


class DeviceManager:
    _nvml_initialized = False
    
    @staticmethod
    def _init_nvml():
        """Initialize NVIDIA Management Library"""
        if not DeviceManager._nvml_initialized and PYNVML_AVAILABLE and pynvml_module is not None:
            try:
                pynvml_module.nvmlInit()
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
    def get_gpu_memory_usage(device_id: int = 0) -> Optional[dict]:
        """Get GPU memory usage for a specific device (in MB) - includes all processes"""
        if not torch.cuda.is_available():
            return None
        
        if PYNVML_AVAILABLE and pynvml_module is not None:
            DeviceManager._init_nvml()
            if DeviceManager._nvml_initialized:
                try:
                    handle = pynvml_module.nvmlDeviceGetHandleByIndex(device_id)
                    mem_info = pynvml_module.nvmlDeviceGetMemoryInfo(handle)
                    total_mb = mem_info.total / 1024 / 1024
                    used_mb = mem_info.used / 1024 / 1024
                    free_mb = mem_info.free / 1024 / 1024
                    
                    return {
                        'allocated_mb': used_mb,
                        'reserved_mb': used_mb,
                        'total_mb': total_mb,
                        'free_mb': free_mb,
                        'usage_percent': (used_mb / total_mb) * 100
                    }
                except Exception as e:
                    logger.warning(f"Failed to get NVML memory info: {e}")
        
        # Fallback to PyTorch memory tracking
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
    def has_sufficient_vram(required_mb: float, device_id: int = 0, safety_margin_mb: float = 2048) -> bool:
        """Check if GPU has sufficient free VRAM"""
        if not torch.cuda.is_available():
            return False
        
        mem_info = DeviceManager.get_gpu_memory_usage(device_id)
        if mem_info is None:
            return False
        
        safe_free = max(0, mem_info['free_mb'] - safety_margin_mb)
        return safe_free >= required_mb
    
    @staticmethod
    def cleanup_gpu_memory(device_id: int = 0) -> Dict:
        """
        GPU memory cleanup for a specific device.
        Returns memory info after cleanup.
        """
        if not torch.cuda.is_available():
            return {}
        
        current_device = torch.cuda.current_device()
        torch.cuda.set_device(device_id)
        
        try:
            # Force Python garbage collection first (multiple rounds)
            for _ in range(5):
                gc.collect()
            
            # Synchronize to ensure all GPU operations complete
            torch.cuda.synchronize(device_id)
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats(device_id)
            torch.cuda.reset_accumulated_memory_stats(device_id)
            
            # Additional synchronization
            torch.cuda.synchronize(device_id)
            
            # One more gc.collect after CUDA ops
            gc.collect()
            
            # Try empty_cache again in case second gc.collect freed more
            torch.cuda.empty_cache()
            
            mem_info = DeviceManager.get_gpu_memory_usage(device_id)
            if mem_info:
                logger.debug(f"GPU {device_id} cleaned - Free: {mem_info['free_mb']:.0f}MB, "
                           f"Usage: {mem_info['usage_percent']:.1f}%")
                return mem_info
            return {}
        finally:
            torch.cuda.set_device(current_device)


class JobDispatcher:
    """
    Dispatches jobs from a queue to available GPUs.
    
    Flow:
    1. Jobs are added to a queue without device assignment
    2. Dispatcher thread monitors queue and GPU availability
    3. When a GPU has sufficient VRAM and is not busy, dispatcher:
       - Acquires the device
       - Dequeues a job
       - Submits the job to a thread pool for execution on that device
       - Releases the device after job completion
    """
    
    def __init__(self, registration_handler, required_vram_mb: float, 
                 vram_check_interval: float, max_workers: int):
        self._registration_handler = registration_handler
        self._required_vram_mb = required_vram_mb
        self._vram_check_interval = vram_check_interval
        
        # Job queue (jobs without device assignment)
        self._job_queue: Queue = Queue()
        
        # Device state tracking
        self._gpu_count = DeviceManager.get_gpu_count()
        self._device_lock = threading.Lock()  # Protects device_busy map
        self._device_busy: Dict[int, bool] = {i: False for i in range(self._gpu_count)}
        
        # Thread pool for executing jobs
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="GPUJobRunner")
        
        # Dispatcher thread
        self._running = True
        self._dispatcher_thread = threading.Thread(target=self._dispatch_loop, daemon=True, name="JobDispatcher")
        self._dispatcher_thread.start()
        
        # Fatal error tracking
        self._fatal_error = None  # Track fatal errors
        self._fatal_error_lock = threading.Lock()
        
        logger.info(f"JobDispatcher started - GPUs: {self._gpu_count}, Max workers: {max_workers}, "
                   f"Required VRAM: {required_vram_mb}MB")
    
    def submit(self, method_name: str, args: tuple, kwargs: dict) -> Future:
        """
        Submit a job to the queue.
        Returns a Future that will contain the result.
        """
        future = Future()
        
        # Extract required_vram_mb from kwargs if provided
        required_vram = kwargs.pop('required_vram_mb', self._required_vram_mb)
        
        job = {
            'method_name': method_name,
            'args': args,
            'kwargs': kwargs,
            'future': future,
            'required_vram_mb': required_vram
        }
        
        self._job_queue.put(job)
        logger.info(f"Job queued: {method_name} (queue size: {self._job_queue.qsize()}, "
                   f"required VRAM: {required_vram}MB)")
        
        return future
    
    def _find_available_device(self, required_vram_mb: float) -> Optional[int]:
        """
        Find a device that is:
        1. Not currently busy (no job running on it)
        2. Has sufficient VRAM available
        
        Returns device_id if found, None otherwise.
        """
        with self._device_lock:
            for device_id in range(self._gpu_count):
                if self._device_busy[device_id]:
                    continue
                
                # Check VRAM before marking as busy
                mem_info = DeviceManager.get_gpu_memory_usage(device_id)
                if mem_info:
                    safe_free = max(0, mem_info['free_mb'] - 2048)  # 2GB safety margin
                    if safe_free >= required_vram_mb:
                        # Mark as busy before returning
                        self._device_busy[device_id] = True
                        logger.info(f"Device {device_id} acquired - Free: {mem_info['free_mb']:.0f}MB, "
                                   f"Required: {required_vram_mb}MB")
                        return device_id
            
        return None
    
    def _release_device(self, device_id: int):
        """Release a device after job completion"""
        with self._device_lock:
            self._device_busy[device_id] = False
            logger.info(f"Device {device_id} released")
    
    def _get_device_status(self) -> Dict[int, Dict]:
        """Get status of all devices"""
        status = {}
        with self._device_lock:
            for device_id in range(self._gpu_count):
                mem_info = DeviceManager.get_gpu_memory_usage(device_id)
                status[device_id] = {
                    'busy': self._device_busy[device_id],
                    'memory': mem_info
                }
        return status
    
    def _run_job_on_device(self, job: Dict, device_id: int):
        """
        Execute a job on a specific device.
        This runs in a thread pool thread.
        """
        method_name = job['method_name']
        args = job['args']
        kwargs = job['kwargs']
        future = job['future']
        
        thread_id = threading.get_ident()
        should_exit = False  # Flag to skip finally cleanup on fatal error
        
        try:
            # Set the device for this thread
            torch.cuda.set_device(device_id)
            logger.info(f"[Thread {thread_id}] Starting job '{method_name}' on GPU {device_id}")
            
            # Get the registration function
            run_fn = getattr(self._registration_handler, method_name, None)
            if run_fn is None:
                raise ValueError(f"Unknown method: {method_name}")
            
            # Execute the job
            result = run_fn(*args, **kwargs)
            
            # Set result on future
            future.set_result(result)
            logger.info(f"[Thread {thread_id}] Completed job '{method_name}' on GPU {device_id}")
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"[Thread {thread_id}] OOM on GPU {device_id}: {e}")
            logger.critical("FATAL: Out of memory - forcing immediate process termination")
            sys.stdout.flush()
            sys.stderr.flush()
            future.set_exception(e)
            should_exit = True
            
            # Kill the entire process immediately
            os.kill(os.getpid(), signal.SIGKILL)
            
        except Exception as e:
            logger.error(f"[Thread {thread_id}] Error on GPU {device_id}: {e}", exc_info=True)
            future.set_exception(e)
            
        finally:
            if not should_exit:
                # Only cleanup if not exiting
                result = None
                args = None
                kwargs = None
                job = dict()
                run_fn = None
                
                gc.collect()
                
                logger.debug(f"[Thread {thread_id}] Cleaning up GPU {device_id}...")
                DeviceManager.cleanup_gpu_memory(device_id)
                
                self._release_device(device_id)
    
    def get_fatal_error(self):
        """Check if a fatal error occurred"""
        with self._fatal_error_lock:
            return self._fatal_error
    
    def has_fatal_error(self) -> bool:
        """Check if dispatcher encountered a fatal error"""
        with self._fatal_error_lock:
            return self._fatal_error is not None
    
    def _dispatch_loop(self):
        """
        Main dispatcher loop:
        1. Peek at the queue to see if there are jobs
        2. Check if any device is available with sufficient VRAM
        3. If yes, dequeue job and submit to thread pool
        4. If no, wait and retry
        """
        logger.info("JobDispatcher loop started")
        
        while self._running:
            try:
                # Check if there are jobs in the queue
                if self._job_queue.empty():
                    time.sleep(0.1)  # Short sleep when queue is empty
                    continue
                
                # Peek at the next job to get its VRAM requirement
                # We need to peek without removing to check device availability first
                try:
                    job = self._job_queue.get_nowait()
                except Empty:
                    continue
                
                required_vram = job['required_vram_mb']
                
                # Try to find an available device
                device_id = self._find_available_device(required_vram)
                
                if device_id is not None:
                    # Device found - submit job to thread pool
                    logger.info(f"Dispatching job '{job['method_name']}' to GPU {device_id}")
                    self._executor.submit(self._run_job_on_device, job, device_id)
                    self._job_queue.task_done()
                else:
                    # No device available - put job back and wait
                    self._job_queue.put(job)
                    
                    # Log device status
                    status = self._get_device_status()
                    for dev_id, dev_status in status.items():
                        busy_str = "BUSY" if dev_status['busy'] else "IDLE"
                        mem = dev_status['memory']
                        if mem:
                            logger.debug(f"GPU {dev_id} [{busy_str}]: Free: {mem['free_mb']:.0f}MB, "
                                       f"Usage: {mem['usage_percent']:.1f}%")
                    
                    logger.debug(f"No device available for job requiring {required_vram}MB. "
                               f"Waiting {self._vram_check_interval}s...")
                    time.sleep(self._vram_check_interval)
                    
            except Exception as e:
                logger.error(f"JobDispatcher error: {e}", exc_info=True)
                time.sleep(1.0)
        
        logger.info("JobDispatcher loop stopped")
    
    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self._job_queue.qsize()
    
    def get_device_status(self) -> Dict[int, Dict]:
        """Get public device status"""
        return self._get_device_status()
    
    def shutdown(self, wait: bool = True):
        """Shutdown the dispatcher and thread pool"""
        logger.info("JobDispatcher shutting down...")
        self._running = False
        
        # Wait for dispatcher thread to stop
        self._dispatcher_thread.join(timeout=5.0)
        
        # Wait for queued jobs to complete
        if wait:
            self._job_queue.join()
        
        # Shutdown thread pool
        self._executor.shutdown(wait=wait)
        
        logger.info("JobDispatcher shutdown complete")


class ProcessJobDispatcher:
    """
    Dispatches jobs to separate processes, each with its own GPU.
    
    Using ProcessPoolExecutor ensures each GPU operation runs in a separate
    process with its own CUDA context, avoiding cuFFT thread-safety issues.
    """
    
    def __init__(self, registration_backend: str, required_vram_mb: float, 
                 vram_check_interval: float, max_workers: int):
        self._registration_backend = registration_backend
        self._required_vram_mb = required_vram_mb
        self._vram_check_interval = vram_check_interval
        
        # Job queue (jobs without device assignment)
        self._job_queue: Queue = Queue()
        
        # Device state tracking
        self._gpu_count = DeviceManager.get_gpu_count()
        self._device_lock = threading.Lock()
        self._device_busy: Dict[int, bool] = {i: False for i in range(self._gpu_count)}
        
        # Pending jobs that couldn't be dispatched (need to retry)
        self._pending_jobs: List[Dict] = []
        self._pending_lock = threading.Lock()
        
        # Process pool - one process per GPU
        # Using 'spawn' context for clean CUDA initialization
        self._mp_context = multiprocessing.get_context('spawn')
        self._executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=self._mp_context
        )
        
        # Thread pool for managing futures and cleanup
        self._future_manager = ThreadPoolExecutor(max_workers=max_workers * 2, 
                                                   thread_name_prefix="FutureManager")
        
        # Dispatcher thread
        self._running = True
        self._dispatcher_thread = threading.Thread(target=self._dispatch_loop, daemon=True, name="ProcessJobDispatcher")
        self._dispatcher_thread.start()
        
        # Fatal error tracking
        self._fatal_error = None
        self._fatal_error_lock = threading.Lock()
        
        logger.info(f"ProcessJobDispatcher started - GPUs: {self._gpu_count}, Max workers: {max_workers}, "
                   f"Required VRAM: {required_vram_mb}MB")
    
    def submit(self, method_name: str, args: tuple, kwargs: dict) -> Future:
        """
        Submit a job to the queue.
        Returns a Future that will contain the result.
        """
        # Create our own future to return to the caller
        user_future = Future()
        
        # Extract required_vram_mb from kwargs if provided
        required_vram = kwargs.pop('required_vram_mb', self._required_vram_mb)
        
        # Validate that args and kwargs are picklable
        try:
            pickle.dumps(args)
            pickle.dumps(kwargs)
        except Exception as e:
            logger.error(f"Job arguments are not picklable: {e}")
            user_future.set_exception(ValueError(f"Job arguments must be picklable for process execution: {e}"))
            return user_future
        
        job = {
            'method_name': method_name,
            'args': args,
            'kwargs': kwargs,
            'user_future': user_future,
            'required_vram_mb': required_vram
        }
        
        self._job_queue.put(job)
        logger.info(f"Job queued: {method_name} (queue size: {self._job_queue.qsize()}, "
                   f"required VRAM: {required_vram}MB)")
        
        return user_future
    
    def _find_available_device(self, required_vram_mb: float) -> Optional[int]:
        """Find a device that is not busy and has sufficient VRAM."""
        with self._device_lock:
            for device_id in range(self._gpu_count):
                if self._device_busy[device_id]:
                    continue
                
                mem_info = DeviceManager.get_gpu_memory_usage(device_id)
                if mem_info:
                    safe_free = max(0, mem_info['free_mb'] - 2048)
                    if safe_free >= required_vram_mb:
                        self._device_busy[device_id] = True
                        logger.info(f"Device {device_id} acquired - Free: {mem_info['free_mb']:.0f}MB, "
                                   f"Required: {required_vram_mb}MB")
                        return device_id
        return None
    
    def _release_device(self, device_id: int):
        """Release a device after job completion"""
        with self._device_lock:
            self._device_busy[device_id] = False
            logger.info(f"Device {device_id} released")
    
    def _get_device_status(self) -> Dict[int, Dict]:
        """Get status of all devices"""
        status = {}
        with self._device_lock:
            for device_id in range(self._gpu_count):
                mem_info = DeviceManager.get_gpu_memory_usage(device_id)
                status[device_id] = {
                    'busy': self._device_busy[device_id],
                    'memory': mem_info
                }
        return status
    
    def _get_available_device_count(self) -> int:
        """Get number of devices that are not busy"""
        with self._device_lock:
            return sum(1 for busy in self._device_busy.values() if not busy)
    
    def _handle_job_completion(self, process_future, user_future: Future, device_id: int, method_name: str):
        """
        Handle completion of a process job.
        Transfers result/exception to user future and releases device.
        """
        try:
            result = process_future.result()
            user_future.set_result(result)
            logger.info(f"Job '{method_name}' completed successfully on GPU {device_id}")
        except Exception as e:
            logger.error(f"Job '{method_name}' failed on GPU {device_id}: {e}")
            user_future.set_exception(e)
        finally:
            self._release_device(device_id)
    
    def _dispatch_job(self, job: Dict) -> bool:
        """
        Try to dispatch a single job to an available device.
        Returns True if dispatched, False if no device available.
        """
        required_vram = job['required_vram_mb']
        device_id = self._find_available_device(required_vram)
        
        if device_id is None:
            return False
        
        method_name = job['method_name']
        user_future = job['user_future']
        
        # Prepare job data for the worker process
        job_data = {
            'device_id': device_id,
            'method_name': method_name,
            'args': job['args'],
            'kwargs': job['kwargs'],
            'registration_backend': self._registration_backend
        }
        
        logger.info(f"Dispatching job '{method_name}' to GPU {device_id} (process)")
        
        # Submit to process pool
        process_future = self._executor.submit(_run_registration_job, job_data)
        
        # Use thread to handle completion callback
        self._future_manager.submit(
            self._handle_job_completion,
            process_future,
            user_future,
            device_id,
            method_name
        )
        
        return True
    
    def _dispatch_loop(self):
        """
        Main dispatcher loop - tries to dispatch multiple jobs in parallel.
        """
        logger.info("ProcessJobDispatcher loop started")
        
        while self._running:
            try:
                jobs_to_dispatch = []
                
                # Collect jobs from pending list first
                with self._pending_lock:
                    jobs_to_dispatch.extend(self._pending_jobs)
                    self._pending_jobs.clear()
                
                # Then collect jobs from queue (up to number of available devices)
                available_devices = self._get_available_device_count()
                while len(jobs_to_dispatch) < available_devices + 2:  # Get a few extra
                    try:
                        job = self._job_queue.get_nowait()
                        jobs_to_dispatch.append(job)
                        self._job_queue.task_done()
                    except Empty:
                        break
                
                if not jobs_to_dispatch:
                    time.sleep(0.05)  # Short sleep when no jobs
                    continue
                
                # Try to dispatch each job
                undispatched = []
                dispatched_count = 0
                
                for job in jobs_to_dispatch:
                    if self._dispatch_job(job):
                        dispatched_count += 1
                    else:
                        undispatched.append(job)
                
                # Put undispatched jobs back in pending list
                if undispatched:
                    with self._pending_lock:
                        self._pending_jobs.extend(undispatched)
                    
                    # Only log and sleep if we couldn't dispatch anything
                    if dispatched_count == 0:
                        status = self._get_device_status()
                        for dev_id, dev_status in status.items():
                            busy_str = "BUSY" if dev_status['busy'] else "IDLE"
                            mem = dev_status['memory']
                            if mem:
                                logger.debug(f"GPU {dev_id} [{busy_str}]: Free: {mem['free_mb']:.0f}MB, "
                                           f"Usage: {mem['usage_percent']:.1f}%")
                        
                        logger.debug(f"No device available. {len(undispatched)} jobs pending. "
                                   f"Waiting {self._vram_check_interval}s...")
                        time.sleep(self._vram_check_interval)
                    else:
                        # Dispatched some jobs, just do a short sleep
                        time.sleep(0.01)
                else:
                    # All jobs dispatched, short sleep
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"ProcessJobDispatcher error: {e}", exc_info=True)
                time.sleep(1.0)
        
        logger.info("ProcessJobDispatcher loop stopped")
    
    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self._job_queue.qsize()
    
    def get_device_status(self) -> Dict[int, Dict]:
        """Get public device status"""
        return self._get_device_status()
    
    def has_fatal_error(self) -> bool:
        """Check if dispatcher encountered a fatal error"""
        with self._fatal_error_lock:
            return self._fatal_error is not None
    
    def shutdown(self, wait: bool = True):
        """Shutdown the dispatcher and process pool"""
        logger.info("ProcessJobDispatcher shutting down...")
        self._running = False
        
        # Wait for dispatcher thread to stop
        self._dispatcher_thread.join(timeout=5.0)
        
        # Wait for queued jobs to complete
        if wait:
            self._job_queue.join()
        
        # Shutdown process pool
        self._executor.shutdown(wait=wait)
        
        # Shutdown future manager
        self._future_manager.shutdown(wait=wait)
        
        logger.info("ProcessJobDispatcher shutdown complete")


class RegistrationManager:
    _instance = None
    _initialized = False
    
    def __new__(cls, registration_backend: Optional[str] = None, number_of_concurrent_runners: Optional[int] = None,
                required_vram_mb: float = 10240, vram_check_interval: float = 5.0, use_processes: bool = True):
        if cls._instance is None:
            cls._instance = super(RegistrationManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, registration_backend: Optional[str] = None, number_of_concurrent_runners: Optional[int] = None,
                 required_vram_mb: float = 10240, vram_check_interval: float = 5.0, use_processes: bool = True):
        if RegistrationManager._initialized:
            return
        
        if registration_backend is None:
            raise ValueError("registration_backend is required for first initialization")
        
        self._registration_backend = registration_backend
        self._registration_handler = None  # Only create in main process for CPU mode
        self._device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._required_vram_mb = required_vram_mb
        self._vram_check_interval = vram_check_interval
        self._use_processes = use_processes
        
        # Determine number of workers
        if number_of_concurrent_runners is None:
            if self._device_type == 'cuda':
                number_of_concurrent_runners = DeviceManager.get_gpu_count()
            else:
                number_of_concurrent_runners = DeviceManager.get_cpu_count()
        
        self._number_of_concurrent_runners = number_of_concurrent_runners
        
        # Setup based on device type and execution mode
        if self._device_type == 'cuda':
            if use_processes:
                # Use ProcessPoolExecutor for CUDA (avoids cuFFT thread-safety issues)
                self._dispatcher = ProcessJobDispatcher(
                    registration_backend=registration_backend,
                    required_vram_mb=required_vram_mb,
                    vram_check_interval=vram_check_interval,
                    max_workers=number_of_concurrent_runners
                )
            else:
                # Legacy thread-based dispatcher (may have cuFFT issues)
                from registration.registration_handler_factory import RegistrationHandlerFactory
                self._registration_handler = RegistrationHandlerFactory.create_registration_handler(registration_backend)
                self._dispatcher = JobDispatcher(
                    registration_handler=self._registration_handler,
                    required_vram_mb=required_vram_mb,
                    vram_check_interval=vram_check_interval,
                    max_workers=number_of_concurrent_runners
                )
        else:
            # CPU: use simple thread pool
            from registration.registration_handler_factory import RegistrationHandlerFactory
            self._registration_handler = RegistrationHandlerFactory.create_registration_handler(registration_backend)
            self._executor = ThreadPoolExecutor(max_workers=self._number_of_concurrent_runners)
        
        RegistrationManager._initialized = True
        mode = "ProcessPool" if (self._device_type == 'cuda' and use_processes) else "ThreadPool"
        logger.info(f"RegistrationManager initialized - Device: {self._device_type}, Mode: {mode}, "
                   f"Workers: {self._number_of_concurrent_runners}, Required VRAM: {self._required_vram_mb}MB")
    
    @staticmethod
    def get_instance():
        """Get the singleton instance of RegistrationManager"""
        if RegistrationManager._instance is None:
            raise ValueError("RegistrationManager is not initialized yet")
        return RegistrationManager._instance
    
    def submit(self, method_name: str, *args, **kwargs) -> Future:
        """
        Submit a registration job.
        
        The job is queued and will be executed when a GPU is available
        with sufficient VRAM.
        
        Args:
            method_name: Name of the registration method to call.
            *args: Positional arguments for the method.
            **kwargs: Keyword arguments for the method.
                - required_vram_mb: Optional VRAM requirement for this job (overrides default)
            
        Returns:
            Future that will contain the result when the job completes.
        """
        if self._device_type == 'cuda':
            return self._dispatcher.submit(method_name, args, kwargs)
        else:
            # CPU: submit directly to thread pool
            run_fn = getattr(self._registration_handler, method_name, None)
            if run_fn is None:
                raise ValueError(f"Unknown method: {method_name}")
            return self._executor.submit(run_fn, *args, **kwargs)
    
    def get_queue_size(self) -> int:
        """Get the current number of jobs in the queue"""
        if self._device_type == 'cuda':
            return self._dispatcher.get_queue_size()
        return 0
    
    def get_device_status(self) -> Dict[int, Dict]:
        """Get status of all GPU devices"""
        if self._device_type == 'cuda':
            return self._dispatcher.get_device_status()
        return {}
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown all workers"""
        if self._device_type == 'cuda':
            self._dispatcher.shutdown(wait=wait)
        else:
            self._executor.shutdown(wait=wait)
        
        logger.info("RegistrationManager shutdown complete")
    
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful for testing)"""
        if cls._instance is not None:
            cls._instance.shutdown(wait=True)
        cls._instance = None
        cls._initialized = False



