from segflow4d.propagation.propagation_strategy.abstract_propagation_strategy import AbstractPropagationStrategy
from segflow4d.common.types.propagation_strategy_name import PropagationStrategyName

class PropagationStrategyFactory:
    @staticmethod
    def create_propagation_strategy(strategy_type: str) -> 'AbstractPropagationStrategy':
        """Create a new propagation strategy based on the strategy type"""
        match strategy_type:
            case PropagationStrategyName.SEQUENTIAL:
                from segflow4d.propagation.propagation_strategy.sequential_propagation_strategy import SequentialPropagationStrategy
                return SequentialPropagationStrategy()
            case PropagationStrategyName.STAR:
                from segflow4d.propagation.propagation_strategy.star_propagation_strategy import StarPropagationStrategy
                return StarPropagationStrategy()
            case _:
                raise ValueError(f"Unknown propagation strategy type: {strategy_type}")