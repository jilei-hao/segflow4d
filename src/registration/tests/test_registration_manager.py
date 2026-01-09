from registration.registration_manager import RegistrationManager, DeviceManager
from registration.registration_handler_factory import RegistrationHandlerFactory
from common.types.registration_methods import REGISTRATION_METHODS
from common.types.registration_backends import REGISTRATION_BACKENDS
from utility.image_helper.image_helper_factory import create_image_helper
import logging

logger = logging.getLogger(__name__)

def main():
    # Configure logging to print to console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example usage of DeviceManager
    device_info = DeviceManager.get_device_info()
    logger.info("Device Information:")
    for key, value in device_info.items():
        logger.info(f"{key}: {value}")

    backend = REGISTRATION_BACKENDS.FIREANTS
    method = REGISTRATION_METHODS.RUN_REGISTRATION_AND_RESLICE

    manager = RegistrationManager(registration_backend=backend)

    # load images
    ih = create_image_helper()
    image_fixed = ih.read_image("/data/jilei/spt/bavcta005/scan2/segflow4d/debug/img_tp-001.nii.gz")
    image_moving = ih.read_image("/data/jilei/spt/bavcta005/scan2/segflow4d/debug/img_tp-002.nii.gz")
    img_to_reslice = ih.read_image("/data/jilei/spt/bavcta005/scan2/seg_ref/sr_tp-001.nii.gz")

    future1 = manager.submit(method, img_fixed=image_fixed, img_moving=image_moving, img_to_reslice=img_to_reslice, mesh_to_reslice={}, options={})
    
    # future3 = manager.submit(method, img_fixed=image_fixed, img_moving=image_moving, img_to_reslice=img_to_reslice, mesh_to_reslice={}, options={})

    # Wait for the job to complete
    try:
        result1 = future1.result()  # Blocks until done
        logger.info(f"Registration 1 completed: {result1}")
        # result3 = future3.result()  # Blocks until done
        # logger.info(f"Registration 3 completed: {result3}")
    except Exception as e:
        logger.error(f"Registration failed: {e}")

    future2 = manager.submit(method, img_fixed=image_fixed, img_moving=image_moving, img_to_reslice=img_to_reslice, mesh_to_reslice={}, options={})

    # Wait for the job to complete
    try:
        result2 = future2.result()  # Blocks until done
        logger.info(f"Registration 2 completed: {result2}")
        # result3 = future3.result()  # Blocks until done
        # logger.info(f"Registration 3 completed: {result3}")
    except Exception as e:
        logger.error(f"Registration failed: {e}")
    finally:
        manager.shutdown(wait=True)

if __name__ == "__main__":
    main()