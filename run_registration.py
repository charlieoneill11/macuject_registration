import wandb
import numpy as np
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
from skimage.transform import warp
from skimage.color import rgb2gray
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nrmse

from retrieve_images import retrieve_images

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

        wandb.init(project='macuject-registration', entity="charlieoneill")
        wandb.log({"L1 loss": np.mean(l1_losses), 
                   "NCC": np.mean(ncc_values), 
                   "SSIM": np.mean(ssim_values)}, 
                   step=wandb.step, run_id=self.registration_function.__name__)
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

    