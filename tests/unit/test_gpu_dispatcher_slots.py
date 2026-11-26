"""ProcessJobDispatcher: per-GPU job slots and event wake-up (sf4d-200 WS-2).

GPU queries are mocked, so this needs no GPU. The process pool starts its workers
lazily and no job is submitted to it, so no worker process is spawned either.
"""

import threading
import time

import pytest

from segflow4d.registration.registration_manager import gpu_registration_manager as grm


@pytest.fixture
def fake_gpu(monkeypatch):
    monkeypatch.setattr(grm.GPUDeviceManager, "get_gpu_count", staticmethod(lambda: 1))
    monkeypatch.setattr(grm.GPUDeviceManager, "get_gpu_memory_usage",
                        staticmethod(lambda device_id: {"free_mb": 8000.0, "used_mb": 0.0,
                                                        "total_mb": 8000.0, "usage_percent": 0.0}))


def _dispatcher(jobs_per_device):
    return grm.ProcessJobDispatcher(registration_backend="fireants", required_vram_mb=0,
                                    vram_check_interval=0.5, max_workers=1,
                                    jobs_per_device=jobs_per_device)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_a_device_takes_exactly_k_jobs(fake_gpu, k):
    d = _dispatcher(k)
    try:
        assert [d._find_available_device(0) for _ in range(k)] == [0] * k
        assert d._find_available_device(0) is None, "a (k+1)-th job must wait"
        d._release_device(0)
        assert d._find_available_device(0) == 0, "a released slot is reusable"
    finally:
        d.shutdown(wait=True)


def test_default_keeps_one_job_per_device(fake_gpu):
    d = grm.ProcessJobDispatcher(registration_backend="fireants", required_vram_mb=0,
                                 vram_check_interval=0.5, max_workers=1)
    try:
        assert d._find_available_device(0) == 0
        assert d._find_available_device(0) is None
    finally:
        d.shutdown(wait=True)


def test_release_wakes_the_dispatcher_at_once(fake_gpu):
    """A pending job must not wait out vram_check_interval after a slot frees up."""
    d = _dispatcher(1)
    try:
        d._find_available_device(0)
        d._wakeup.clear()
        woke = []
        waiter = threading.Thread(target=lambda: woke.append(d._wakeup.wait(timeout=2.0)))
        waiter.start()
        time.sleep(0.05)
        t0 = time.time()
        d._release_device(0)
        waiter.join()
        assert woke == [True] and time.time() - t0 < 0.2
    finally:
        d.shutdown(wait=True)

