from abc import ABC, abstractmethod
from common.types.propagation_strategy_data import PropagationStrategyData

class AbstractPropagationStrategy(ABC):
    @abstractmethod
    def propagate(self, tp_input_data: dict[int, PropagationStrategyData], options) -> dict[int, PropagationStrategyData]:
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        pass