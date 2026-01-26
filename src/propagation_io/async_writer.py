import threading
import queue
import time
from typing import Callable, Any, NamedTuple, Optional
from common.types.image_wrapper import ImageWrapper
from common.types.mesh_wrapper import MeshWrapper
from propagation_io.image_writers import sitk_image_writer
from propagation_io.mesh_writers import vtk_polydata_writer
from logging import getLogger

logger = getLogger(__name__)

# Writer signature: writer(data, filename)
WriterFn = Callable[[Any, str], None]


class _Task(NamedTuple):
    writer: WriterFn
    data: Any
    filename: str


class AsyncWriter:
    def __init__(self, max_queue: int = 0):
        self._q: queue.Queue[Optional[_Task]] = queue.Queue(maxsize=max_queue)
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    def submit(self, writer: WriterFn, data: Any, filename: str) -> None:
        """Enqueue a write task; returns immediately."""
        self._q.put(_Task(writer=writer, data=data, filename=filename))
        logger.info(f"AsyncWriter: Submitted write task for {filename}")

    def submit_image(self, image: ImageWrapper, filename: str) -> None:
        """Enqueue a write task for an image using the default image writer."""
        
        sitk_image = image.get_data()
        self._q.put(_Task(writer=sitk_image_writer, data=sitk_image, filename=filename))
        logger.info(f"AsyncWriter: Submitted image write task for {filename}")
        
    def submit_mesh(self, mesh: MeshWrapper, filename: str) -> None:
        """Enqueue a write task for a mesh using the default mesh writer."""
        polydata = mesh.get_data()
        self._q.put(_Task(writer=vtk_polydata_writer, data=polydata, filename=filename))
        logger.info(f"AsyncWriter: Submitted mesh write task for {filename}")

    def shutdown(self, wait: bool = True) -> None:
        """Signal the worker to exit."""
        self._q.put(None)
        if wait:
            self._t.join()

    def _worker(self) -> None:
        while True:
            task = self._q.get()
            if task is None:
                self._q.task_done()
                break
            try:
                logger.info(f"AsyncWriter: Writing file {task.filename}")
                start_time = time.time()
                task.writer(task.data, task.filename)
                elapsed_time = time.time() - start_time
                logger.debug(f"AsyncWriter: Completed write task for {task.filename} (took {elapsed_time:.3f}s)")
            except Exception as e:
                logger.error(f"Error writing file {task.filename}: {e}")
            finally:
                # Drop references to allow GC
                task = None
                self._q.task_done()


# Global singleton
async_writer = AsyncWriter()