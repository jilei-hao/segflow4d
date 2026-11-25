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
    # Sequential low-res mask propagation, then SASD at HIGH resolution.
    #
    # sequential_star and sasd_star both run a plain star at high res: every
    # target is registered to the reference directly, with no initialisation.
    # That is fine a frame or two out, but on bavcta025 accuracy collapses with
    # phase distance (Dice 0.92 at +-1 frame, 0.27 at +12) while the low-res
    # mask keeps tracking — i.e. the images are registrable frame-to-frame and
    # it is the direct ref->target jump that fails. This combo gives the
    # high-res stage the phase-chained affine initialiser SASD was written for.
    SEQUENTIAL_SASD = "sequential_sasd"
    # Direct: skip the low-res mask propagation phase entirely and run
    # StarDirectRegistrationStrategy at high resolution. Useful as a
    # baseline when comparing against the full SegFlow4D pipeline.
    DIRECT_STAR = "direct_star"