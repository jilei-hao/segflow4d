from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class AbstractRegistrationOptions(ABC):
    '''
    A class representing abstract registration options.
    '''


    @abstractmethod
    def __post_init__(self):
        pass