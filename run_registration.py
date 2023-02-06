import numpy as np
from pathlib import Path
import os
import cv2
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
from skimage.transform import warp
from skimage.color import rgb2gray
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nrmse

import argparse

parser = argparse.ArgumentParser(description='Script to handle command line arguments')
parser.add_argument('--registration_algo', type=str, default="optical_flow", help='Algorithm to use for registration')
args = parser.parse_args()

def retrieve_images(patient_id = '156518'):
    # Set the root directory for the patient data
    root_dir = Path(f'data/{patient_id}')

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


class RegistrationAlgorithm:
    
    def __init__(self, registration_function):
        self.registration_function = registration_function
        self.final_images, self.template = retrieve_images()
        self.registered_images = self.apply_registration()
        
    def apply_registration(self):
        # Do the registration process
        registered_images = []
        for i, img in enumerate(tqdm(self.final_images)):
            registered = self.registration_function(self.template, img) 
            registered_images.append(registered)
        return registered_images
    
    def evaluate_registration(self):
        l1_losses = []
        ncc_values = []
        ssim_values = []

        for registered_img in self.registered_images:
            l1_loss = np.mean(np.abs(self.template - registered_img))
            l1_losses.append(l1_loss)

            ncc = np.corrcoef(self.template.ravel(), registered_img.ravel())[0,1]
            ncc_values.append(ncc)

            ssim_value = ssim(self.template, registered_img, data_range=registered_img.max() - registered_img.min())
            ssim_values.append(ssim_value)

        return l1_losses, ncc_values, ssim_values


### REGISTRATION ALGORITHMS ###
def optical_flow(template, img):
    # calculate the vector field for optical flow
    v, u = optical_flow_tvl1(template, img)
    # use the estimated optical flow for registration
    nr, nc = template.shape
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc),
                                         indexing='ij')
    registered = warp(img, np.array([row_coords + v, col_coords + u]), mode='edge')
    return registered

### MAPPINGS ###
algo_dict = {'optical_flow': optical_flow}
 

if __name__ == "__main__":
    opt = RegistrationAlgorithm(algo_dict[args.registration_algo])
    l1_losses, ncc_values, ssim_values = opt.evaluate_registration()
    print("L1 losses:", f"{np.mean(l1_losses):.2f}")
    print("Normalized cross-correlation values:", f"{np.mean(ncc_values):.2f}")
    print("Structural similarity index values:", f"{np.mean(ssim_values):.2f}")
    