from .cpu_image_helper import CPUImageHelper
from .abstract_image_helper import AbstractImageHelper

def create_image_helper() -> AbstractImageHelper:
    return CPUImageHelper()
