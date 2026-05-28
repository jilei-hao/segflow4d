class PropagationStrategyName:
    SEQUENTIAL = "SequentialPropagationStrategy"
    STAR = "StarPropagationStrategy"
    SASD = "SASDPropagationStrategy"
    STAR_DIRECT = "StarDirectRegistrationStrategy"


class PropagationStrategyCombo:
    """Valid values for ``PropagationOptions.propagation_strategy_combo``.

    A combo selects which low-res strategy (if any) precedes the high-res
    ref → target registration star.
    """
    SEQUENTIAL_STAR = "sequential_star"
    SASD_STAR = "sasd_star"
    # Direct: skip the low-res mask propagation phase entirely and run
    # StarDirectRegistrationStrategy at high resolution. Useful as a
    # baseline when comparing against the full SegFlow4D pipeline.
    DIRECT_STAR = "direct_star"