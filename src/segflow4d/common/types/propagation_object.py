from abc import ABC, abstractmethod

class PropagationObject(ABC):
    def __init__(self):
        self.object_name = "UnnamedPropagationObject"

    def set_object_name(self, name: str):
        self.object_name = name
