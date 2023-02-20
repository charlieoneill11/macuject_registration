import re
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import pandas as pd
from tqdm.notebook import tqdm
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
from skimage.transform import warp
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nrmse


def retrieve_images(patient_id = '156518', laterality = 'L', date = None):
    # Set the root directory for the patient data
    root_dir = Path(f'../data/{patient_id}')

    # Get the list of image filenames for the left eye
    image_filenames = [f for f in os.listdir(root_dir) if f'{laterality}.png' in f]
    
    # If we are registering to same visit, only keep files from given date
    if date != None:
        pattern = re.compile(r"\w+_(\d{4}-\d{2}-\d{2})_")
        image_filenames = [file for file in image_filenames if date in file]

    # Read the images into a list
    images = [cv2.imread(str(root_dir / f)) for f in image_filenames]

    # Convert the images to grayscale
    gray_images = [rgb2gray(img) for img in images]

    # Register all images to the first image
    template = gray_images[0]

    # Remove invalid images
    final_images = [x for x in gray_images[1:] if x.shape == template.shape]

    return final_images, template

def evaluate_registration(template_img: np.ndarray, 
                          registered_imgs: List[np.ndarray]) -> (List[float], List[float], List[float]):
    """
    Evaluate the registration quality of multiple registered images with respect to a template image.
    """
    l1_losses = []
    ncc_values = []
    ssim_values = []
    
    for registered_img in registered_imgs:
        # Compute L1 loss between the template and registered images
        l1_loss = np.mean(np.abs(template_img - registered_img))
        l1_losses.append(l1_loss)
        
        # Compute normalized cross-correlation between the template and registered images
        ncc = np.corrcoef(template_img.ravel(), registered_img.ravel())[0,1]
        ncc_values.append(ncc)
        
        # Compute structural similarity index between the template and registered images
        ssim_value = ssim(template_img, registered_img, data_range=registered_img.max() - registered_img.min())
        ssim_values.append(ssim_value)
        
    return l1_losses, ncc_values, ssim_values

def visualise_registration_results(registered_images, original_images, template, loss_values):
    num_images = min(len(registered_images), 3)
    
    # Get the indices of the three images with the highest L1 losses
    top_indices = np.argsort(loss_values)[-num_images:]

    # Get the indices of the three images with the lowest L1 losses
    bottom_indices = np.argsort(loss_values)[:num_images]

    # Create the grid figure
    fig, axes = plt.subplots(num_images, 4, figsize=(20, 15))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    # Loop through the top three images
    for i, idx in enumerate(top_indices):
        # Plot the original image in the first column of the left section
        ax = axes[i][0]
        ax.imshow(original_images[idx], cmap='gray')
        original_l1 = np.mean(np.abs(template - original_images[idx]))
        ax.set_title("Original Image (L1 Loss: {:.2f})".format(original_l1))

        # Plot the registered image in the second column of the left section
        ax = axes[i][1]
        ax.imshow(registered_images[idx], cmap='gray')
        ax.set_title("Registered Image (L1 Loss: {:.2f})".format(loss_values[idx]))

    # Loop through the bottom three images
    for i, idx in enumerate(bottom_indices):
        # Plot the original image in the first column of the right section
        ax = axes[i][2]
        ax.imshow(original_images[idx], cmap='gray')
        original_l1 = np.mean(np.abs(template - original_images[idx]))
        ax.set_title("Original Image (L1 Loss: {:.2f})".format(original_l1))

        # Plot the registered image in the second column of the right section
        ax = axes[i][3]
        ax.imshow(registered_images[idx], cmap='gray')
        ax.set_title("Registered Image (L1 Loss: {:.2f})".format(loss_values[idx]))

    # Show the grid
    plt.show()
    
def highlight_worse(val, comparison_column, worse_val, better_val):
    color = better_val if val == worse_val else worse_val
    return 'background-color: {}'.format(color)

def style_df(df_dict):
    df = pd.DataFrame(df_dict)
    for column in df.columns:
        comparison_column = 'original' if column == 'registered' else 'registered'
        worse_val = 'red'
        better_val = 'green'
        if column in ['ncc', 'ssim']:
            worse_val, better_val = better_val, worse_val
        df.style.apply(highlight_worse, axis=1, subset=[column], comparison_column=comparison_column, worse_val=worse_val, better_val=better_val)
    return df

def summarise_registration(original_images, registered_images, template):
    
    # Calculate metrics for original images
    l1_losses, ncc_values, ssim_values = evaluate_registration(template, original_images)
    l1_original, ncc_original, ssim_original = np.mean(l1_losses), np.mean(ncc_values), np.mean(ssim_values)
    
    # Calculate metrics for registered images
    l1_losses, ncc_values, ssim_values = evaluate_registration(template, registered_images)
    l1_registered, ncc_registered, ssim_registered = np.mean(l1_losses), np.mean(ncc_values), np.mean(ssim_values)
    
    # Create dataframe
    df_dict = {'original': {'l1': l1_original, 'ncc': ncc_original, 'ssim': ssim_original}, 
               'registered': {'l1': l1_registered, 'ncc': ncc_registered, 'ssim': ssim_registered}}
    
    return style_df(df_dict)

def extract_dates(filenames):
    date_pattern = re.compile(r'\d{8}')
    dates = []
    for filename in filenames:
        date = date_pattern.search(filename)
        if date:
            dates.append(date.group())
    return dates

# Set the root directory for the patient data
root_dir = Path(f'../data/156518')

# Get the list of image filenames for the left eye
image_filenames = [f for f in os.listdir(root_dir) if 'L.png' in f]

dates = extract_dates(image_filenames)
counts = dict(Counter(dates))
sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))