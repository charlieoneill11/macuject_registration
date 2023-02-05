from pathlib import Path
from skimage.color import rgb2gray
import os
import cv2


def retrieve_images(patient_id = '156518'):
    # Set the root directory for the patient data
    root_dir = Path(f'../data/{patient_id}')

    # Get the list of image filenames for the left eye
    image_filenames = [f for f in os.listdir(root_dir) if 'L.png' in f]

    # Read the images into a list
    images = [cv2.imread(str(root_dir / f)) for f in image_filenames]

    # Convert the images to grayscale
    gray_images = [rgb2gray(img) for img in images]

    # Register all images to the first image
    template = gray_images[0]

    # Remove invalid images
    final_images = [x for x in gray_images[1:] if x.shape == template.shape]

    return final_images, template