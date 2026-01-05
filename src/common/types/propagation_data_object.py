from abc import ABC, abstractmethod

from common.types.propagation_object import PropagationObject
from logging import getLogger

logger = getLogger(__name__)

class PropagationDataObject(PropagationObject):
    def __init__(self):
        super().__init__()
        self._data = None

    @abstractmethod
    def get_data(self):
        return self._data

    @abstractmethod
    def set_data(self, data):
        self._data = data


    def __del__(self):
        try:
            if hasattr(self, '_data') and self._data is not None:
                self._data = None
        except Exception as e:
            logger.error(f"Error during ImageWrapper cleanup: {e}")
