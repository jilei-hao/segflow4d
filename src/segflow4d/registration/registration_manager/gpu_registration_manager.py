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

import numpy as np
import torch

from segflow4d.registration.registration_manager.abstract_registration_manager import AbstractRegistrationManager
from segflow4d.registration.registration_manager.gpu_device_manager import GPUDeviceManager
from segflow4d.registration.registration_manager.gpu_worker import (
    JobOutcome,
    run_registration_job,
    persistent_gpu_worker,
)
from segflow4d.utility import device_utils

logger = logging.getLogger(__name__)


# Head-room a job needs on an idle device beyond its required_vram_mb, as before
# (it was a literal 2048 in each dispatcher). The rule for the first job is unchanged.
VRAM_MARGIN_MB = 2048.0
# Head-room kept free when packing further jobs next to running ones. It is smaller
# than VRAM_MARGIN_MB because the reservations come from measured peaks (allocator
# plus CUDA context), not from an instantaneous free-memory reading.
CONCURRENT_VRAM_MARGIN_MB = 512.0
# A measured per-job peak is scaled by this before it is reserved for the next such job.
VRAM_ESTIMATE_SAFETY = 1.10
# Extra factor when an estimate is extrapolated from a same-method job of another size.
VRAM_SCALED_ESTIMATE_SAFETY = 1.25


def _is_out_of_memory(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "out of memory" in text or "outofmemory" in text


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
        self._device_busy: dict[int, bool] = {i: False for i in range(self._gpu_count)}
        
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
    
    def _find_available_device(self, required_vram_mb: float) -> int | None:
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
    
    def _get_device_status(self) -> dict[int, dict]:
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
    
    def _run_job_on_device(self, job: dict, device_id: int):
        """Execute a job on a specific device."""
        method_name = job['method_name']
        args = job['args']
        kwargs = job['kwargs']
        future = job['future']
        
        thread_id = threading.get_ident()
        should_exit = False
        
        try:
            device_utils.set_device(device_id)
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
    
    def get_device_status(self) -> dict[int, dict]:
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

        # We deliberately do NOT call _job_queue.join() here. The dispatch
        # loop can re-queue jobs without a matching task_done(), so the
        # unfinished-tasks counter can drift above zero even after every
        # dispatched job completes. Waiting on the executor below is the
        # real correctness guarantee — any still-queued (never-dispatched)
        # jobs are dropped intentionally on shutdown.
        self._executor.shutdown(wait=wait)
        logger.info("ThreadJobDispatcher shutdown complete")


class ProcessJobDispatcher:
    """
    Process-based job dispatcher for GPU jobs.
    
    Using ProcessPoolExecutor ensures each GPU operation runs in a separate
    process with its own CUDA context, avoiding cuFFT thread-safety issues.
    """
    
    def __init__(self, registration_backend: str, required_vram_mb: float, 
                 vram_check_interval: float, max_workers: int, jobs_per_device: int = 1):
        self._registration_backend = registration_backend
        self._required_vram_mb = required_vram_mb
        self._vram_check_interval = vram_check_interval
        
        self._job_queue: Queue = Queue()
        self._gpu_count = GPUDeviceManager.get_gpu_count()
        self._device_lock = threading.Lock()
        # Jobs running per device. Up to jobs_per_device run at once on one GPU, each
        # in its own worker process. A job spends much of its time off the GPU (IPC,
        # host-side prep, result packaging), and another job's GPU work can fill that.
        self._jobs_per_device = max(1, int(jobs_per_device))
        self._device_jobs: dict[int, int] = {i: 0 for i in range(self._gpu_count)}
        # Memory-aware admission (only used when jobs_per_device > 1). A second job
        # starts only if the VRAM reserved by the running jobs plus its own estimate
        # fits the device's budget. The budget is the free VRAM, less a margin,
        # measured when the device was idle, so other processes' usage is accounted
        # for. Estimates are learned per kind of job (method + fixed-image size) from
        # the peaks the workers report. The first job of an unknown kind runs alone.
        # So small (cropped) jobs run side by side, and a large, barely cropped
        # volume still runs one at a time on a moderate card.
        self._device_reserved_mb: dict[int, float] = {i: 0.0 for i in range(self._gpu_count)}
        self._device_budget_mb: dict[int, float] = {i: 0.0 for i in range(self._gpu_count)}
        self._device_exclusive: dict[int, bool] = {i: False for i in range(self._gpu_count)}
        self._device_draining: dict[int, bool] = {i: False for i in range(self._gpu_count)}
        self._job_vram_mb: dict[tuple, float] = {}
        # Woken on submit and on release, so a waiting job is dispatched the moment
        # a slot frees up, rather than after a fixed vram_check_interval sleep (0.5 s
        # from the pipeline: about 0.25 s of idle GPU per job).
        self._wakeup = threading.Event()
        
        self._pending_jobs: list[dict] = []
        self._pending_lock = threading.Lock()
        
        self._mp_context = mp.get_context('spawn')
        self._executor = ProcessPoolExecutor(
            max_workers=max_workers * self._jobs_per_device,
            mp_context=self._mp_context
        )
        
        self._future_manager = ThreadPoolExecutor(max_workers=max_workers * self._jobs_per_device * 2, 
                                                   thread_name_prefix="FutureManager")
        
        self._running = True
        self._dispatcher_thread = threading.Thread(target=self._dispatch_loop, daemon=True, name="ProcessJobDispatcher")
        self._dispatcher_thread.start()
        
        self._fatal_error = None
        self._fatal_error_lock = threading.Lock()
        
        logger.info(f"ProcessJobDispatcher started - GPUs: {self._gpu_count}, Max workers: {max_workers}, "
                   f"jobs per GPU: {self._jobs_per_device}, Required VRAM: {required_vram_mb}MB")
    
    def submit(self, method_name: str, args: tuple, kwargs: dict) -> Future:
        """Submit a job to the queue."""
        user_future = Future()
        required_vram = kwargs.pop('required_vram_mb', self._required_vram_mb)
        # No dry-run pickle here: the process pool pickles the arguments to send them
        # anyway, and a pickling failure reaches user_future through the process
        # future. A dry run serialised every job's images twice on the submitting thread.
        
        job = {
            'method_name': method_name,
            'args': args,
            'kwargs': kwargs,
            'user_future': user_future,
            'required_vram_mb': required_vram,
            'key': self._job_key(method_name, kwargs),
            'exclusive': False,
        }
        
        self._job_queue.put(job)
        logger.info(f"Job queued: {method_name} (queue size: {self._job_queue.qsize()}, "
                   f"required VRAM: {required_vram}MB)")
        self._wakeup.set()
        
        return user_future
    
    @staticmethod
    def _job_key(method_name: str, kwargs: dict) -> tuple:
        """Jobs with the same method and fixed-image size are assumed to need the same VRAM."""
        img = kwargs.get('img_fixed')
        try:
            size = tuple(img.get_data().GetSize()) if img is not None else None
        except Exception:
            size = None
        return (method_name, size)

    def _estimate_mb(self, job: dict) -> float | None:
        """VRAM to reserve for a job, or None if it must run alone (unknown kind, or a retry).

        A measured peak for the same kind is used directly. Otherwise the estimate is
        extrapolated from the largest measured job of the same method by voxel count.
        Smaller jobs just take that job's peak, which is conservative because the CUDA
        context does not shrink. A method never seen before runs alone once to be measured.
        """
        if job.get('exclusive'):
            return None
        method, size = job['key']
        learned = self._job_vram_mb.get(job['key'])
        if learned is not None:
            estimate = learned * VRAM_ESTIMATE_SAFETY
        else:
            same_method = [(k[1], peak) for k, peak in self._job_vram_mb.items()
                           if k[0] == method and k[1] is not None]
            if size is None or not same_method:
                return None
            ref_size, ref_peak = max(same_method, key=lambda kv: kv[1])
            ratio = float(np.prod(size)) / float(np.prod(ref_size))
            estimate = ref_peak * max(1.0, ratio) * VRAM_SCALED_ESTIMATE_SAFETY
        return max(estimate, float(job['required_vram_mb']))

    def _find_available_device(self, job: dict) -> int | None:
        """Admit a job to a device with a free slot and room for it; None if it must wait.

        With jobs_per_device == 1 this is the original rule: an idle device whose
        free VRAM, less the margin, covers the job's required_vram_mb.
        """
        estimate = self._estimate_mb(job) if self._jobs_per_device > 1 else 0.0
        with self._device_lock:
            for device_id in range(self._gpu_count):
                running = self._device_jobs[device_id]
                if running >= self._jobs_per_device:
                    continue

                if running == 0:
                    mem_info = GPUDeviceManager.get_gpu_memory_usage(device_id)
                    if not mem_info:
                        continue
                    safe_free = max(0.0, mem_info['free_mb'] - VRAM_MARGIN_MB)
                    if safe_free < job['required_vram_mb']:
                        continue
                    self._device_budget_mb[device_id] = max(
                        0.0, mem_info['free_mb'] - CONCURRENT_VRAM_MARGIN_MB)
                    self._device_draining[device_id] = False
                    exclusive = self._jobs_per_device > 1 and estimate is None
                    self._device_exclusive[device_id] = exclusive
                    job['reservation_mb'] = 0.0 if exclusive else (estimate or 0.0)
                else:
                    # Joining jobs already running: only with a known estimate that fits.
                    if self._device_exclusive[device_id] or self._device_draining[device_id]:
                        continue
                    if estimate is None:
                        # Must run alone: stop admitting others here until the device drains.
                        self._device_draining[device_id] = True
                        logger.info(f"Device {device_id} draining for a job that runs alone "
                                    f"({job['method_name']}, {job['key'][1]})")
                        continue
                    if self._device_reserved_mb[device_id] + estimate > self._device_budget_mb[device_id]:
                        continue
                    job['reservation_mb'] = estimate

                self._device_jobs[device_id] += 1
                self._device_reserved_mb[device_id] += job['reservation_mb']
                logger.info(
                    f"Device {device_id} acquired ({self._device_jobs[device_id]}/{self._jobs_per_device} slots"
                    + (", alone" if self._device_exclusive[device_id] else
                       f", reserved {self._device_reserved_mb[device_id]:.0f}/"
                       f"{self._device_budget_mb[device_id]:.0f} MiB" if self._jobs_per_device > 1 else "")
                    + f") - Required: {job['required_vram_mb']}MB")
                return device_id
        return None

    def _release_device(self, device_id: int, job: dict | None = None, peak_vram_mb: float | None = None):
        """Release one job slot on a device after job completion, and learn its VRAM peak."""
        with self._device_lock:
            self._device_jobs[device_id] = max(0, self._device_jobs[device_id] - 1)
            if job is not None:
                self._device_reserved_mb[device_id] = max(
                    0.0, self._device_reserved_mb[device_id] - job.get('reservation_mb', 0.0))
                if peak_vram_mb is not None:
                    key = job['key']
                    self._job_vram_mb[key] = max(self._job_vram_mb.get(key, 0.0), peak_vram_mb)
            if self._device_jobs[device_id] == 0:
                self._device_exclusive[device_id] = False
                self._device_reserved_mb[device_id] = 0.0
            logger.info(f"Device {device_id} released")
        self._wakeup.set()

    def _get_device_status(self) -> dict[int, dict]:
        """Get status of all devices."""
        status = {}
        with self._device_lock:
            for device_id in range(self._gpu_count):
                mem_info = GPUDeviceManager.get_gpu_memory_usage(device_id)
                status[device_id] = {
                    'busy': self._device_jobs[device_id] >= self._jobs_per_device,
                    'memory': mem_info
                }
        return status
    
    def _get_available_device_count(self) -> int:
        """Get number of devices that are not busy."""
        with self._device_lock:
            return sum(self._jobs_per_device - n for n in self._device_jobs.values())
    
    def _handle_job_completion(self, process_future, job: dict, device_id: int):
        """Handle completion of a process job: deliver the result, learn the VRAM peak,
        and retry a job alone once if it ran out of GPU memory next to other jobs."""
        method_name, user_future = job['method_name'], job['user_future']
        try:
            outcome = process_future.result()
        except Exception as e:
            self._release_device(device_id, job)
            if self._jobs_per_device > 1 and not job.get('exclusive') and _is_out_of_memory(e):
                logger.warning(f"Job '{method_name}' ran out of GPU memory alongside other jobs "
                               f"on GPU {device_id}; retrying it alone")
                job['exclusive'] = True
                with self._pending_lock:
                    self._pending_jobs.append(job)
                self._wakeup.set()
                return
            logger.error(f"Job '{method_name}' failed on GPU {device_id}: {e}")
            user_future.set_exception(e)
            return
        peak = outcome.peak_vram_mb if isinstance(outcome, JobOutcome) else None
        result = outcome.result if isinstance(outcome, JobOutcome) else outcome
        self._release_device(device_id, job, peak)
        user_future.set_result(result)
        logger.info(f"Job '{method_name}' completed successfully on GPU {device_id}"
                    + (f" (peak {peak:.0f} MiB)" if peak is not None else ""))

    def _dispatch_job(self, job: dict) -> bool:
        """Try to dispatch a single job to an available device."""
        device_id = self._find_available_device(job)
        
        if device_id is None:
            return False
        
        method_name = job['method_name']
        
        job_data = {
            'device_id': device_id,
            'method_name': method_name,
            'args': job['args'],
            'kwargs': job['kwargs'],
            'registration_backend': self._registration_backend,
            'log_level': logging.getLogger().level or logging.DEBUG
        }
        
        logger.info(f"Dispatching job '{method_name}' to GPU {device_id} (process)")
        
        process_future = self._executor.submit(run_registration_job, job_data)
        
        self._future_manager.submit(
            self._handle_job_completion,
            process_future,
            job,
            device_id,
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
                    self._wakeup.wait(0.05)
                    self._wakeup.clear()
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
                                   f"Waiting up to {self._vram_check_interval}s...")
                        # Woken at once when a slot is released or a job arrives; the
                        # timeout only paces re-checks when VRAM (not a slot) is short.
                        self._wakeup.wait(self._vram_check_interval)
                        self._wakeup.clear()
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
    
    def get_device_status(self) -> dict[int, dict]:
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
            self._executor.shutdown(wait=True)
        else:
            # Take the worker list BEFORE shutting the pool down: shutdown() sets
            # _processes to None. Iterating it afterwards always raised AttributeError,
            # which callers such as the CLI's SIGTERM handler swallowed, so workers
            # (and the GPU memory they hold) outlived a killed run
            # (sf4d-200 KNOWN_ISSUES B5).
            workers = list((self._executor._processes or {}).values())
            self._executor.shutdown(wait=False, cancel_futures=True)
            for proc in workers:
                try:
                    proc.terminate()
                except Exception:
                    pass

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
        self._job_queues: dict[int, MPQueue] = {}
        self._result_queues: dict[int, MPQueue] = {}
        self._workers: dict[int, BaseProcess] = {}
        self._pending_futures: dict[int, Future] = {}
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
        self._device_busy: dict[int, bool] = {i: False for i in range(self._gpu_count)}
        
        self._job_queue: Queue = Queue()
        self._pending_jobs: list[dict] = []
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
    
    def _find_available_device(self, required_vram_mb: float) -> int | None:
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
    
    def _get_device_status(self) -> dict[int, dict]:
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
    
    def get_device_status(self) -> dict[int, dict]:
        """Get public device status."""
        return self._get_device_status()
    
    def shutdown(self, wait: bool = True):
        """Shutdown all workers."""
        self._running = False

        if wait:
            for job_queue in self._job_queues.values():
                job_queue.put(None)
            for worker in self._workers.values():
                worker.join(timeout=5.0)
        else:
            # Terminate worker processes immediately
            for worker in self._workers.values():
                try:
                    if worker.is_alive():
                        worker.terminate()
                except Exception:
                    pass


class GPURegistrationManager(AbstractRegistrationManager):
    """
    GPU-based registration manager.
    
    Manages GPU resources and dispatches registration jobs to available GPUs.
    Uses process-based execution by default to avoid CUDA thread-safety issues.
    """
    
    def __init__(self, registration_backend: str,
                 max_workers: int | None = None,
                 required_vram_mb: float = 10240,
                 vram_check_interval: float = 0.1,
                 use_processes: bool = True,
                 use_persistent_workers: bool = False,
                 jobs_per_device: int = 1):
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
            jobs_per_device: Jobs run concurrently on each GPU (process pool only).
        """
        if not device_utils.is_accelerator_available():
            raise RuntimeError(
                "No GPU accelerator (CUDA or MPS) is available. "
                "Use CPURegistrationManager instead."
            )

        device_kind = device_utils.detect_device_kind()
        self._registration_backend = registration_backend
        self._device_type = device_kind
        self._required_vram_mb = required_vram_mb
        self._vram_check_interval = vram_check_interval

        if max_workers is None:
            max_workers = GPUDeviceManager.get_gpu_count() or 1
        self._max_workers = max_workers

        # MPS cannot survive fork (no shared device context across processes)
        # and has no inter-process CUDA-context issue to avoid, so always use
        # the in-process ThreadJobDispatcher on MPS.
        if device_kind == "mps" and (use_persistent_workers or use_processes):
            logger.info(
                "MPS backend detected — overriding subprocess dispatcher with "
                "ThreadJobDispatcher (fork+MPS is unsupported)."
            )
            use_persistent_workers = False
            use_processes = False

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
                max_workers=max_workers,
                jobs_per_device=jobs_per_device,
            )
            mode = "ProcessPool"
        else:
            from segflow4d.registration.registration_handler import RegistrationHandlerFactory
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
    
    def get_device_status(self) -> dict[int, dict]:
        """Get status of all GPU devices."""
        return self._dispatcher.get_device_status()
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the manager."""
        logger.info("GPURegistrationManager shutting down...")
        self._dispatcher.shutdown(wait=wait)
        logger.info("GPURegistrationManager shutdown complete")
