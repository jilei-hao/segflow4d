from segflow4d.common.types.abstract_registration_options import AbstractRegistrationOptions
from dataclasses import dataclass, field
from typing import Any, Literal

@dataclass
class FireantsRegistrationOptions(AbstractRegistrationOptions):
    """
    Registration options for FireANTs registration pipeline.
    
    Attributes:
        scales: Multi-resolution scales for registration (coarse to fine)
        affine_iterations: Number of iterations per scale for affine stage
        deformable_iterations: Number of iterations per scale for deformable stage
        affine_lr: Learning rate for affine registration optimizer
        deformable_lr: Learning rate for deformable registration optimizer
        loss_type: Similarity metric to use ('mse' or 'cc' for cross-correlation)
        cc_kernel_size: Kernel size for cross-correlation loss (only used if loss_type='cc')
        deformation_type: Type of deformation model ('compositive' or 'additive')
        smooth_grad_sigma_mm: Gaussian smoothing sigma for gradients during optimization (in millimeters)
        smooth_warp_sigma_mm: Gaussian smoothing sigma for warp field regularization (in millimeters)
        torch_dtype: Accepted from config but currently ignored (fused ops require float32).
    """
    
    scales: list[float] = field(default_factory=lambda: [4, 2, 1])
    affine_iterations: list[int] = field(default_factory=lambda: [200, 100, 50])
    deformable_iterations: list[int] = field(default_factory=lambda: [200, 100, 25])
    affine_lr: float = 3e-3
    deformable_lr: float = 0.5
    loss_type: Literal['mse', 'cc'] = 'mse'
    cc_kernel_size: int = 3
    deformation_type: Literal['compositive', 'additive'] = 'compositive'
    smooth_grad_sigma_mm: float = 3.0
    smooth_warp_sigma_mm: float = 1.5
    torch_dtype: Any = None  # accepted from config but ignored (fused ops require float32)
    
    def __post_init__(self):
        """Validate registration options after initialization."""
        if not self.scales or len(self.scales) == 0:
            raise ValueError("scales must be a non-empty list")
        
        if len(self.affine_iterations) != len(self.scales):
            raise ValueError(f"affine_iterations length ({len(self.affine_iterations)}) must match scales length ({len(self.scales)})")
        
        if len(self.deformable_iterations) != len(self.scales):
            raise ValueError(f"deformable_iterations length ({len(self.deformable_iterations)}) must match scales length ({len(self.scales)})")
        
        if self.affine_lr <= 0:
            raise ValueError("affine_lr must be positive")
        
        if self.deformable_lr <= 0:
            raise ValueError("deformable_lr must be positive")
        
        if self.loss_type not in ['mse', 'cc']:
            raise ValueError(f"loss_type must be 'mse' or 'cc', got '{self.loss_type}'")
        
        if self.cc_kernel_size <= 0:
            raise ValueError("cc_kernel_size must be positive")
        
        if self.deformation_type not in ['compositive', 'additive']:
            raise ValueError(f"deformation_type must be 'compositive' or 'additive', got '{self.deformation_type}'")
        
        if self.smooth_grad_sigma_mm < 0:
            raise ValueError("smooth_grad_sigma_mm must be non-negative")
        
        if self.smooth_warp_sigma_mm < 0:
            raise ValueError("smooth_warp_sigma_mm must be non-negative")