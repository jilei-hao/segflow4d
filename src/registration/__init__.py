"""
Registration Package

This package provides image registration functionality including:
- Registration handlers for different backends (FireANTs, Greedy)
- Registration managers for CPU and GPU execution
"""

# Re-export main classes for backward compatibility
from registration.registration_manager import (
    RegistrationManager,
    RegistrationManagerFactory,
    AbstractRegistrationManager,
    CPURegistrationManager,
    GPURegistrationManager,
    GPUDeviceManager,
)

from registration.registration_handler import (
    AbstractRegistrationHandler,
    RegistrationHandlerFactory,
)

__all__ = [
    # Manager classes
    'RegistrationManager',
    'RegistrationManagerFactory',
    'AbstractRegistrationManager',
    'CPURegistrationManager',
    'GPURegistrationManager',
    'GPUDeviceManager',
    # Handler classes
    'AbstractRegistrationHandler',
    'RegistrationHandlerFactory',
]
