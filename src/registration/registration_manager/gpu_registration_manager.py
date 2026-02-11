"""
GPU-based registration manager using process-based job dispatching.

This module provides GPU registration management with proper CUDA context
isolation using separate processes for each GPU.
"""

import gc
import logging
import multiprocessing as mp
import os
import pickle
import signal
import sys
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Queue as MPQueue
from multiprocessing.process import BaseProcess
from queue import Empty, Queue
from typing import Dict, List, Optional

import torch

from registration.registration_manager.abstract_registration_manager import AbstractRegistrationManager
from registration.registration_manager.gpu_device_manager import GPUDeviceManager
from registration.registration_manager.gpu_worker import (
    run_registration_job,
    persistent_gpu_worker,
)

logger = logging.getLogger(__name__)


class ThreadJobDispatcher:
    """
    Thread-based job dispatcher for GPU jobs.
    
    Uses ThreadPoolExecutor which may have cuFFT thread-safety issues
    with some registration backends. Consider using ProcessJobDispatcher
    for better stability.
    
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
        self._gpu_count = GPUDeviceManager.get_gpu_count()
        self._device_lock = threading.Lock()
        self._device_busy: Dict[int, bool] = {i: False for i in range(self._gpu_count)}
        
        # Thread pool for executing jobs
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="GPUJobRunner")
        
        # Dispatcher thread
        self._running = True
        self._dispatcher_thread = threading.Thread(target=self._dispatch_loop, daemon=True, name="ThreadJobDispatcher")
        self._dispatcher_thread.start()
        
        # Fatal error tracking
        self._fatal_error = None
        self._fatal_error_lock = threading.Lock()
        
        logger.info(f"ThreadJobDispatcher started - GPUs: {self._gpu_count}, Max workers: {max_workers}, "
                   f"Required VRAM: {required_vram_mb}MB")
    
    def submit(self, method_name: str, args: tuple, kwargs: dict) -> Future:
        """Submit a job to the queue."""
        future = Future()
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
        """Find a device that is not busy and has sufficient VRAM."""
        with self._device_lock:
            for device_id in range(self._gpu_count):
                if self._device_busy[device_id]:
                    continue
                
                mem_info = GPUDeviceManager.get_gpu_memory_usage(device_id)
                if mem_info:
                    safe_free = max(0, mem_info['free_mb'] - 2048)
                    if safe_free >= required_vram_mb:
                        self._device_busy[device_id] = True
                        logger.info(f"Device {device_id} acquired - Free: {mem_info['free_mb']:.0f}MB, "
                                   f"Required: {required_vram_mb}MB")
                        return device_id
                    else:
                        logger.info(f"Device {device_id} insufficient VRAM - Free: {mem_info['free_mb']:.0f}MB, "
                                     f"Required: {required_vram_mb}MB")
        return None
    
    def _release_device(self, device_id: int):
        """Release a device after job completion."""
        with self._device_lock:
            self._device_busy[device_id] = False
            logger.info(f"Device {device_id} released")
    
    def _get_device_status(self) -> Dict[int, Dict]:
        """Get status of all devices."""
        status = {}
        with self._device_lock:
            for device_id in range(self._gpu_count):
                mem_info = GPUDeviceManager.get_gpu_memory_usage(device_id)
                status[device_id] = {
                    'busy': self._device_busy[device_id],
                    'memory': mem_info
                }
        return status
    
    def _run_job_on_device(self, job: Dict, device_id: int):
        """Execute a job on a specific device."""
        method_name = job['method_name']
        args = job['args']
        kwargs = job['kwargs']
        future = job['future']
        
        thread_id = threading.get_ident()
        should_exit = False
        
        try:
            torch.cuda.set_device(device_id)
            logger.info(f"[Thread {thread_id}] Starting job '{method_name}' on GPU {device_id}")
            
            run_fn = getattr(self._registration_handler, method_name, None)
            if run_fn is None:
                raise ValueError(f"Unknown method: {method_name}")
            
            result = run_fn(*args, **kwargs)
            future.set_result(result)
            logger.info(f"[Thread {thread_id}] Completed job '{method_name}' on GPU {device_id}")
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"[Thread {thread_id}] OOM on GPU {device_id}: {e}")
            logger.critical("FATAL: Out of memory - forcing immediate process termination")
            sys.stdout.flush()
            sys.stderr.flush()
            future.set_exception(e)
            should_exit = True
            os.kill(os.getpid(), signal.SIGKILL)
            
        except Exception as e:
            logger.error(f"[Thread {thread_id}] Error on GPU {device_id}: {e}", exc_info=True)
            future.set_exception(e)
            
        finally:
            if not should_exit:
                gc.collect()
                logger.debug(f"[Thread {thread_id}] Cleaning up GPU {device_id}...")
                GPUDeviceManager.cleanup_gpu_memory(device_id)
                self._release_device(device_id)
    
    def _dispatch_loop(self):
        """Main dispatcher loop."""
        logger.info("ThreadJobDispatcher loop started")
        
        while self._running:
            try:
                if self._job_queue.empty():
                    time.sleep(0.1)
                    continue
                
                try:
                    job = self._job_queue.get_nowait()
                except Empty:
                    continue
                
                required_vram = job['required_vram_mb']
                device_id = self._find_available_device(required_vram)
                
                if device_id is not None:
                    logger.info(f"Dispatching job '{job['method_name']}' to GPU {device_id}")
                    self._executor.submit(self._run_job_on_device, job, device_id)
                    self._job_queue.task_done()
                else:
                    self._job_queue.put(job)
                    
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
                logger.error(f"ThreadJobDispatcher error: {e}", exc_info=True)
                time.sleep(1.0)
        
        logger.info("ThreadJobDispatcher loop stopped")
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self._job_queue.qsize()
    
    def get_device_status(self) -> Dict[int, Dict]:
        """Get public device status."""
        return self._get_device_status()
    
    def has_fatal_error(self) -> bool:
        """Check if dispatcher encountered a fatal error."""
        with self._fatal_error_lock:
            return self._fatal_error is not None
    
    def shutdown(self, wait: bool = True):
        """Shutdown the dispatcher and thread pool."""
        logger.info("ThreadJobDispatcher shutting down...")
        self._running = False
        self._dispatcher_thread.join(timeout=5.0)
        
        if wait:
            self._job_queue.join()
        
        self._executor.shutdown(wait=wait)
        logger.info("ThreadJobDispatcher shutdown complete")


class ProcessJobDispatcher:
    """
    Process-based job dispatcher for GPU jobs.
    
    Using ProcessPoolExecutor ensures each GPU operation runs in a separate
    process with its own CUDA context, avoiding cuFFT thread-safety issues.
    """
    
    def __init__(self, registration_backend: str, required_vram_mb: float, 
                 vram_check_interval: float, max_workers: int):
        self._registration_backend = registration_backend
        self._required_vram_mb = required_vram_mb
        self._vram_check_interval = vram_check_interval
        
        self._job_queue: Queue = Queue()
        self._gpu_count = GPUDeviceManager.get_gpu_count()
        self._device_lock = threading.Lock()
        self._device_busy: Dict[int, bool] = {i: False for i in range(self._gpu_count)}
        
        self._pending_jobs: List[Dict] = []
        self._pending_lock = threading.Lock()
        
        self._mp_context = mp.get_context('spawn')
        self._executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=self._mp_context
        )
        
        self._future_manager = ThreadPoolExecutor(max_workers=max_workers * 2, 
                                                   thread_name_prefix="FutureManager")
        
        self._running = True
        self._dispatcher_thread = threading.Thread(target=self._dispatch_loop, daemon=True, name="ProcessJobDispatcher")
        self._dispatcher_thread.start()
        
        self._fatal_error = None
        self._fatal_error_lock = threading.Lock()
        
        logger.info(f"ProcessJobDispatcher started - GPUs: {self._gpu_count}, Max workers: {max_workers}, "
                   f"Required VRAM: {required_vram_mb}MB")
    
    def submit(self, method_name: str, args: tuple, kwargs: dict) -> Future:
        """Submit a job to the queue."""
        user_future = Future()
        required_vram = kwargs.pop('required_vram_mb', self._required_vram_mb)
        
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
                
                mem_info = GPUDeviceManager.get_gpu_memory_usage(device_id)
                if mem_info:
                    safe_free = max(0, mem_info['free_mb'] - 2048)
                    if safe_free >= required_vram_mb:
                        self._device_busy[device_id] = True
                        logger.info(f"Device {device_id} acquired - Free: {mem_info['free_mb']:.0f}MB, "
                                   f"Required: {required_vram_mb}MB")
                        return device_id
        return None
    
    def _release_device(self, device_id: int):
        """Release a device after job completion."""
        with self._device_lock:
            self._device_busy[device_id] = False
            logger.info(f"Device {device_id} released")
    
    def _get_device_status(self) -> Dict[int, Dict]:
        """Get status of all devices."""
        status = {}
        with self._device_lock:
            for device_id in range(self._gpu_count):
                mem_info = GPUDeviceManager.get_gpu_memory_usage(device_id)
                status[device_id] = {
                    'busy': self._device_busy[device_id],
                    'memory': mem_info
                }
        return status
    
    def _get_available_device_count(self) -> int:
        """Get number of devices that are not busy."""
        with self._device_lock:
            return sum(1 for busy in self._device_busy.values() if not busy)
    
    def _handle_job_completion(self, process_future, user_future: Future, device_id: int, method_name: str):
        """Handle completion of a process job."""
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
        """Try to dispatch a single job to an available device."""
        required_vram = job['required_vram_mb']
        device_id = self._find_available_device(required_vram)
        
        if device_id is None:
            return False
        
        method_name = job['method_name']
        user_future = job['user_future']
        
        job_data = {
            'device_id': device_id,
            'method_name': method_name,
            'args': job['args'],
            'kwargs': job['kwargs'],
            'registration_backend': self._registration_backend
        }
        
        logger.info(f"Dispatching job '{method_name}' to GPU {device_id} (process)")
        
        process_future = self._executor.submit(run_registration_job, job_data)
        
        self._future_manager.submit(
            self._handle_job_completion,
            process_future,
            user_future,
            device_id,
            method_name
        )
        
        return True
    
    def _dispatch_loop(self):
        """Main dispatcher loop - tries to dispatch multiple jobs in parallel."""
        logger.info("ProcessJobDispatcher loop started")
        
        while self._running:
            try:
                jobs_to_dispatch = []
                
                with self._pending_lock:
                    jobs_to_dispatch.extend(self._pending_jobs)
                    self._pending_jobs.clear()
                
                available_devices = self._get_available_device_count()
                while len(jobs_to_dispatch) < available_devices + 2:
                    try:
                        job = self._job_queue.get_nowait()
                        jobs_to_dispatch.append(job)
                        self._job_queue.task_done()
                    except Empty:
                        break
                
                if not jobs_to_dispatch:
                    time.sleep(0.05)
                    continue
                
                undispatched = []
                dispatched_count = 0
                
                for job in jobs_to_dispatch:
                    if self._dispatch_job(job):
                        dispatched_count += 1
                    else:
                        undispatched.append(job)
                
                if undispatched:
                    with self._pending_lock:
                        self._pending_jobs.extend(undispatched)
                    
                    if dispatched_count == 0:
                        status = self._get_device_status()
                        for dev_id, dev_status in status.items():
                            busy_str = "BUSY" if dev_status['busy'] else "IDLE"
                            mem = dev_status['memory']
                            if mem:
                                logger.debug(f"GPU {dev_id} [{busy_str}]: Free: {mem['free_mb']:.0f}MB, "
                                           f"Usage: {mem['usage_percent']:.1f}%, required: {self._required_vram_mb}MB")
                        
                        logger.debug(f"No device available. {len(undispatched)} jobs pending. "
                                   f"Waiting {self._vram_check_interval}s...")
                        time.sleep(self._vram_check_interval)
                    else:
                        time.sleep(0.01)
                else:
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"ProcessJobDispatcher error: {e}", exc_info=True)
                time.sleep(1.0)
        
        logger.info("ProcessJobDispatcher loop stopped")
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self._job_queue.qsize()
    
    def get_device_status(self) -> Dict[int, Dict]:
        """Get public device status."""
        return self._get_device_status()
    
    def has_fatal_error(self) -> bool:
        """Check if dispatcher encountered a fatal error."""
        with self._fatal_error_lock:
            return self._fatal_error is not None
    
    def shutdown(self, wait: bool = True):
        """Shutdown the dispatcher and process pool."""
        logger.info("ProcessJobDispatcher shutting down...")
        self._running = False
        self._dispatcher_thread.join(timeout=5.0)
        
        if wait:
            self._job_queue.join()
        
        self._executor.shutdown(wait=wait)
        self._future_manager.shutdown(wait=wait)
        logger.info("ProcessJobDispatcher shutdown complete")


class PersistentProcessJobDispatcher:
    """
    Dispatcher using persistent worker processes per GPU.
    
    Avoids process spawn overhead by keeping worker processes alive
    and reusing them for multiple jobs.
    """
    
    def __init__(self, registration_backend: str, required_vram_mb: float,
                 vram_check_interval: float, max_workers: int):
        self._registration_backend = registration_backend
        self._required_vram_mb = required_vram_mb
        self._vram_check_interval = min(vram_check_interval, 0.1)
        
        self._gpu_count = GPUDeviceManager.get_gpu_count()
        self._job_counter = 0
        self._job_counter_lock = threading.Lock()
        
        self._mp_context = mp.get_context('spawn')
        self._job_queues: Dict[int, MPQueue] = {}
        self._result_queues: Dict[int, MPQueue] = {}
        self._workers: Dict[int, BaseProcess] = {}
        self._pending_futures: Dict[int, Future] = {}
        self._pending_futures_lock = threading.Lock()
        
        for device_id in range(min(max_workers, self._gpu_count)):
            job_queue = self._mp_context.Queue()
            result_queue = self._mp_context.Queue()
            
            worker = self._mp_context.Process(
                target=persistent_gpu_worker,
                args=(device_id, registration_backend, job_queue, result_queue),
                daemon=True
            )
            worker.start()
            
            self._job_queues[device_id] = job_queue
            self._result_queues[device_id] = result_queue
            self._workers[device_id] = worker
        
        self._device_lock = threading.Lock()
        self._device_busy: Dict[int, bool] = {i: False for i in range(self._gpu_count)}
        
        self._job_queue: Queue = Queue()
        self._pending_jobs: List[Dict] = []
        self._pending_lock = threading.Lock()
        
        self._running = True
        self._dispatcher_thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._dispatcher_thread.start()
        
        self._result_threads = []
        for device_id in range(min(max_workers, self._gpu_count)):
            t = threading.Thread(target=self._result_collector, args=(device_id,), daemon=True)
            t.start()
            self._result_threads.append(t)
        
        logger.info(f"PersistentProcessJobDispatcher started - GPUs: {self._gpu_count}")
    
    def _result_collector(self, device_id: int):
        """Collect results from a specific GPU worker."""
        result_queue = self._result_queues[device_id]
        
        while self._running:
            try:
                try:
                    result_data = result_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                job_id = result_data['job_id']
                result = result_data['result']
                error = result_data['error']
                
                with self._pending_futures_lock:
                    future = self._pending_futures.pop(job_id, None)
                
                if future:
                    if error:
                        future.set_exception(error)
                    else:
                        future.set_result(result)
                
                with self._device_lock:
                    self._device_busy[device_id] = False
                    logger.info(f"Device {device_id} released")
                    
            except Exception as e:
                logger.error(f"Result collector error for GPU {device_id}: {e}")
    
    def submit(self, method_name: str, args: tuple, kwargs: dict) -> Future:
        """Submit a job."""
        user_future = Future()
        required_vram = kwargs.pop('required_vram_mb', self._required_vram_mb)
        
        with self._job_counter_lock:
            job_id = self._job_counter
            self._job_counter += 1
        
        job = {
            'job_id': job_id,
            'method_name': method_name,
            'args': args,
            'kwargs': kwargs,
            'user_future': user_future,
            'required_vram_mb': required_vram
        }
        
        self._job_queue.put(job)
        return user_future
    
    def _find_available_device(self, required_vram_mb: float) -> Optional[int]:
        """Find available device."""
        with self._device_lock:
            for device_id in self._workers.keys():
                if self._device_busy[device_id]:
                    continue
                
                mem_info = GPUDeviceManager.get_gpu_memory_usage(device_id)
                if mem_info and mem_info['free_mb'] - 1024 >= required_vram_mb:
                    self._device_busy[device_id] = True
                    return device_id
        return None
    
    def _get_device_status(self) -> Dict[int, Dict]:
        """Get status of all devices."""
        status = {}
        with self._device_lock:
            for device_id in range(self._gpu_count):
                mem_info = GPUDeviceManager.get_gpu_memory_usage(device_id)
                status[device_id] = {
                    'busy': self._device_busy[device_id],
                    'memory': mem_info
                }
        return status
    
    def _dispatch_loop(self):
        """Dispatch jobs to available GPU workers."""
        while self._running:
            try:
                jobs_to_dispatch = []
                
                with self._pending_lock:
                    jobs_to_dispatch.extend(self._pending_jobs)
                    self._pending_jobs.clear()
                
                while True:
                    try:
                        job = self._job_queue.get_nowait()
                        jobs_to_dispatch.append(job)
                        self._job_queue.task_done()
                    except Empty:
                        break
                
                if not jobs_to_dispatch:
                    time.sleep(0.05)
                    continue
                
                undispatched = []
                for job in jobs_to_dispatch:
                    device_id = self._find_available_device(job['required_vram_mb'])
                    
                    if device_id is not None:
                        job_id = job['job_id']
                        
                        with self._pending_futures_lock:
                            self._pending_futures[job_id] = job['user_future']
                        
                        job_data = {
                            'job_id': job_id,
                            'method_name': job['method_name'],
                            'args': job['args'],
                            'kwargs': job['kwargs']
                        }
                        self._job_queues[device_id].put(job_data)
                        logger.info(f"Dispatched job {job_id} to GPU {device_id}")
                    else:
                        undispatched.append(job)
                
                if undispatched:
                    with self._pending_lock:
                        self._pending_jobs.extend(undispatched)
                    time.sleep(0.1)
                else:
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Dispatch error: {e}")
                time.sleep(1.0)
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self._job_queue.qsize()
    
    def get_device_status(self) -> Dict[int, Dict]:
        """Get public device status."""
        return self._get_device_status()
    
    def shutdown(self, wait: bool = True):
        """Shutdown all workers."""
        self._running = False
        
        for device_id, job_queue in self._job_queues.items():
            job_queue.put(None)
        
        if wait:
            for worker in self._workers.values():
                worker.join(timeout=5.0)


class GPURegistrationManager(AbstractRegistrationManager):
    """
    GPU-based registration manager.
    
    Manages GPU resources and dispatches registration jobs to available GPUs.
    Uses process-based execution by default to avoid CUDA thread-safety issues.
    """
    
    def __init__(self, registration_backend: str,
                 max_workers: Optional[int] = None,
                 required_vram_mb: float = 10240,
                 vram_check_interval: float = 0.1,
                 use_processes: bool = True,
                 use_persistent_workers: bool = False):
        """
        Initialize GPU registration manager.
        
        Args:
            registration_backend: Backend to use ('fireants', 'greedy', etc.)
            max_workers: Maximum number of concurrent GPU workers.
                        Defaults to GPU count.
            required_vram_mb: Required VRAM per job in MB.
            vram_check_interval: Interval for checking VRAM availability.
            use_processes: Use process-based execution (recommended).
            use_persistent_workers: Use persistent worker processes (reduces overhead).
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Use CPURegistrationManager instead.")
        
        self._registration_backend = registration_backend
        self._device_type = 'cuda'
        self._required_vram_mb = required_vram_mb
        self._vram_check_interval = vram_check_interval
        
        if max_workers is None:
            max_workers = GPUDeviceManager.get_gpu_count()
        self._max_workers = max_workers
        
        # Select dispatcher type
        if use_persistent_workers:
            self._dispatcher = PersistentProcessJobDispatcher(
                registration_backend=registration_backend,
                required_vram_mb=required_vram_mb,
                vram_check_interval=vram_check_interval,
                max_workers=max_workers
            )
            mode = "PersistentProcess"
        elif use_processes:
            self._dispatcher = ProcessJobDispatcher(
                registration_backend=registration_backend,
                required_vram_mb=required_vram_mb,
                vram_check_interval=vram_check_interval,
                max_workers=max_workers
            )
            mode = "ProcessPool"
        else:
            from registration.registration_handler import RegistrationHandlerFactory
            registration_handler = RegistrationHandlerFactory.create_registration_handler(registration_backend)
            self._dispatcher = ThreadJobDispatcher(
                registration_handler=registration_handler,
                required_vram_mb=required_vram_mb,
                vram_check_interval=vram_check_interval,
                max_workers=max_workers
            )
            mode = "ThreadPool"
        
        logger.info(f"GPURegistrationManager initialized - Mode: {mode}, "
                   f"Workers: {max_workers}, Required VRAM: {required_vram_mb}MB")
    
    def submit(self, method_name: str, *args, **kwargs) -> Future:
        """Submit a registration job."""
        return self._dispatcher.submit(method_name, args, kwargs)
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self._dispatcher.get_queue_size()
    
    def get_device_status(self) -> Dict[int, Dict]:
        """Get status of all GPU devices."""
        return self._dispatcher.get_device_status()
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the manager."""
        logger.info("GPURegistrationManager shutting down...")
        self._dispatcher.shutdown(wait=wait)
        logger.info("GPURegistrationManager shutdown complete")
