from dataclasses import dataclass

@dataclass
class PropagationOptions:
    '''
    A class representing options for propagation.
    '''
    lowres_resample_factor: float
    dilation_radius: int