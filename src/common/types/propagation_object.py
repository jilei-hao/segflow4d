from abc import ABC, abstractmethod

class PropagationObject(ABC):
    def __init__(self):
        self.object_name = "UnnamedPropagationObject"

    def set_object_name(self, name: str):
        self.object_name = name

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
    
