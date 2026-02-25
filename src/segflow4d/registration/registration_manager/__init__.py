"""
Registration Manager Package

This package provides registration job management for both CPU and GPU execution.
"""

from segflow4d.registration.registration_manager.abstract_registration_manager import AbstractRegistrationManager
from segflow4d.registration.registration_manager.cpu_registration_manager import CPURegistrationManager
from segflow4d.registration.registration_manager.gpu_registration_manager import GPURegistrationManager
from segflow4d.registration.registration_manager.gpu_device_manager import GPUDeviceManager
from segflow4d.registration.registration_manager.factory import RegistrationManagerFactory, RegistrationManager

__all__ = [
    'AbstractRegistrationManager',
    'CPURegistrationManager',
    'GPURegistrationManager',
    'GPUDeviceManager',
    'RegistrationManagerFactory',
    'RegistrationManager',
]
