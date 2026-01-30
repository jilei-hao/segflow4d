from dataclasses import dataclass, field
from typing import Any

@dataclass
class PropagationOptions:
    '''
    A class representing options for propagation.
    '''
    lowres_resample_factor: float
    dilation_radius: int
    registration_backend: str
    registration_backend_options: dict[str, Any] = field(default_factory=dict) # options specific to the registration backend
    write_result_to_disk: bool = False
    output_directory: str = ""
    debug: bool = False
    debug_output_directory: str = ""
    minimum_required_vram_gb: int = 10

    def __post_init__(self):
        pass