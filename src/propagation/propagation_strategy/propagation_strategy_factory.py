from propagation.propagation_strategy.abstract_propagation_strategy import AbstractPropagationStrategy
from common.types.propagation_strategy_name import PropagationStrategyName

class PropagationStrategyFactory:
    @staticmethod
    def create_propagation_strategy(strategy_type: str) -> 'AbstractPropagationStrategy':
        """Create a new propagation strategy based on the strategy type"""
        if strategy_type == PropagationStrategyName.SEQUENTIAL:
            from propagation.propagation_strategy.sequential_propagation_strategy import SequentialPropagationStrategy
            return SequentialPropagationStrategy()
        elif strategy_type == PropagationStrategyName.STAR:
            from propagation.propagation_strategy.star_propagation_strategy import StarPropagationStrategy
            return StarPropagationStrategy()
        else:
            raise ValueError(f"Unknown propagation strategy type: {strategy_type}")