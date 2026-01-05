from SimpleITK import Image, WriteImage

def sitk_image_writer(image: Image, file_path: str):
    WriteImage(image, file_path)


