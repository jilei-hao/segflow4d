from common.types.propagation_options import PropagationOptions
import SimpleITK as sitk


class PropagationPipeline:
    def __init__(self, options: PropagationOptions):
        self._options = options

    def setInputs(self, image4d: sitk.Image, seg_groups: list)


    def _prepare_data(self):
        pass


    def _propagate(self):
        pass