"""ProcessJobDispatcher: per-GPU job slots, memory-aware admission, event wake-up (sf4d-200 WS-2).

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


def _job(method="run_registration_and_reslice", size=(64, 64, 64), required_mb=0):
    """A queued job as submit() builds it, without the images."""
    return {"method_name": method, "required_vram_mb": required_mb,
            "key": (method, size), "exclusive": False}


def _dispatcher(jobs_per_device):
    return grm.ProcessJobDispatcher(registration_backend="fireants", required_vram_mb=0,
                                    vram_check_interval=0.5, max_workers=1,
                                    jobs_per_device=jobs_per_device)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_a_device_takes_at_most_k_small_jobs(fake_gpu, k):
    d = _dispatcher(k)
    d._job_vram_mb[_job()["key"]] = 500.0  # a known, small kind of job
    try:
        jobs = [_job() for _ in range(k + 1)]
        assert [d._find_available_device(j) for j in jobs[:k]] == [0] * k
        assert d._find_available_device(jobs[k]) is None, "a (k+1)-th job must wait"
        d._release_device(0, jobs[0])
        assert d._find_available_device(jobs[k]) == 0, "a released slot is reusable"
    finally:
        d.shutdown(wait=False)


def test_default_keeps_one_job_per_device(fake_gpu):
    d = grm.ProcessJobDispatcher(registration_backend="fireants", required_vram_mb=0,
                                 vram_check_interval=0.5, max_workers=1)
    try:
        assert d._find_available_device(_job()) == 0
        assert d._find_available_device(_job()) is None
    finally:
        d.shutdown(wait=False)


def test_unknown_kind_runs_alone_then_shares_once_its_peak_is_known(fake_gpu):
    d = _dispatcher(3)
    try:
        probe, other = _job(), _job()
        assert d._find_available_device(probe) == 0
        assert d._find_available_device(other) is None, "nothing joins a probe"
        d._release_device(0, probe, peak_vram_mb=1400.0)  # the worker's measured peak
        assert d._job_vram_mb[probe["key"]] == 1400.0
        assert [d._find_available_device(_job()) for _ in range(3)] == [0, 0, 0]
    finally:
        d.shutdown(wait=False)


def test_large_jobs_run_one_at_a_time_on_a_moderate_card(fake_gpu):
    """The case gpu_jobs_per_device must survive: a barely cropped volume. The fake
    card has 8000 MiB free, so the concurrency budget is 8000 - 512 = 7488 MiB. A job
    peaking at 4000 MiB (4400 MiB reserved with the safety factor) leaves no room
    for a second."""
    d = _dispatcher(3)
    big = _job(size=(512, 512, 300))
    d._job_vram_mb[big["key"]] = 4000.0
    try:
        first, second = _job(size=(512, 512, 300)), _job(size=(512, 512, 300))
        assert d._find_available_device(first) == 0
        assert d._find_available_device(second) is None, "two big jobs would not fit"
        d._release_device(0, first)
        assert d._find_available_device(second) == 0
    finally:
        d.shutdown(wait=False)


def test_small_jobs_fill_the_budget_and_no_further(fake_gpu):
    d = _dispatcher(8)
    d._job_vram_mb[_job()["key"]] = 2000.0  # 2200 MiB reserved each; budget 7488
    try:
        admitted = [d._find_available_device(_job()) for _ in range(5)]
        assert admitted == [0, 0, 0, None, None], "three fit (6600 MiB), a fourth would not"
    finally:
        d.shutdown(wait=False)


def test_a_job_that_must_run_alone_drains_the_device(fake_gpu):
    d = _dispatcher(3)
    d._job_vram_mb[_job()["key"]] = 500.0
    try:
        running = _job()
        assert d._find_available_device(running) == 0
        loner = _job(method="run_deformable_and_reslice")  # unknown kind: must run alone
        assert d._find_available_device(loner) is None
        assert d._find_available_device(_job()) is None, "no new company while draining"
        d._release_device(0, running)
        assert d._find_available_device(loner) == 0
    finally:
        d.shutdown(wait=False)


def test_out_of_memory_next_to_other_jobs_is_retried_alone(fake_gpu):
    from concurrent.futures import Future
    d = _dispatcher(2)
    try:
        job = _job()
        job["user_future"] = Future()
        assert d._find_available_device(job) == 0
        failed = Future()
        failed.set_exception(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
        d._handle_job_completion(failed, job, 0)
        assert not job["user_future"].done(), "the caller must not see the OOM"
        assert job["exclusive"] and job in d._pending_jobs, "requeued to run alone"
        assert d._device_jobs[0] == 0
    finally:
        d.shutdown(wait=False)


def test_out_of_memory_when_already_alone_is_a_real_failure(fake_gpu):
    from concurrent.futures import Future
    d = _dispatcher(2)
    try:
        job = _job()
        job["user_future"] = Future()
        job["exclusive"] = True
        d._find_available_device(job)
        failed = Future()
        failed.set_exception(RuntimeError("CUDA out of memory"))
        d._handle_job_completion(failed, job, 0)
        with pytest.raises(RuntimeError, match="out of memory"):
            job["user_future"].result(timeout=1)
    finally:
        d.shutdown(wait=False)


def test_release_wakes_the_dispatcher_at_once(fake_gpu):
    """A pending job must not wait out vram_check_interval after a slot frees up."""
    d = _dispatcher(1)
    try:
        d._find_available_device(_job())
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
        d.shutdown(wait=False)


def test_shutdown_without_wait_does_not_raise(fake_gpu):
    """It used to iterate the pool's worker table after shutdown() had set it to None."""
    _dispatcher(2).shutdown(wait=False)


def test_shutdown_without_wait_terminates_live_workers(fake_gpu):
    """sf4d-200 KNOWN_ISSUES B5: a killed run must not leave GPU workers behind."""
    import os
    d = _dispatcher(1)
    pid = d._executor.submit(os.getpid).result(timeout=60)  # starts one spawn worker
    d.shutdown(wait=False)
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    pytest.fail(f"worker {pid} still alive 10 s after shutdown(wait=False)")


def test_same_method_other_size_is_extrapolated_not_probed(fake_gpu):
    """Each direction crops its own box, so high-res sizes differ; they must not all
    run alone as probes. A known 1000 MiB job of 100^3 voxels gives a 125^3 job about
    1000 * 1.95 * 1.25 = 2441 MiB, and a smaller one keeps the known 1000 MiB peak."""
    d = _dispatcher(3)
    d._job_vram_mb[("run_registration_and_reslice", (100, 100, 100))] = 1000.0
    try:
        bigger = _job(size=(125, 125, 125))
        smaller = _job(size=(80, 80, 80))
        assert d._estimate_mb(bigger) == pytest.approx(1000.0 * 125**3 / 100**3 * 1.25)
        assert d._estimate_mb(smaller) == pytest.approx(1000.0 * 1.25)
        assert d._estimate_mb(_job(method="run_affine_only")) is None, "a new method is probed"
        assert [d._find_available_device(j) for j in (bigger, smaller)] == [0, 0]
    finally:
        d.shutdown(wait=False)
