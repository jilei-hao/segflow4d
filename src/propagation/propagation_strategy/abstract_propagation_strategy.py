from abc import ABC, abstractmethod
from common.types.tp_data import TPData

class AbstractPropagationStrategy(ABC):
    @abstractmethod
    def propagate(self, tp_input_data: dict[int, TPData], options) -> dict[int, TPData]:
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        pass