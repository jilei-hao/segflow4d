import warnings

try:
    import fireants_fused_ops  # noqa: F401
except ImportError:
    warnings.warn(
        "FireANTs fused CUDA ops are not installed. "
        "Registration performance will be degraded. "
        "Run 'segflow4d-install-fireants' to build and install them.",
        ImportWarning,
        stacklevel=2,
    )
