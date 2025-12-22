from dataclasses import dataclass, field
from typing import Any

@dataclass
class PropagationOptions:
    '''
    A class representing options for propagation.
    '''
    lowres_resample_factor: float
    dilation_radius: int
    registration_backend_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        pass