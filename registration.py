import os
import re
import numpy as np
import cv2

def extract_image_data(file_name):
    # extract patient id, visit date, fundus id, and laterality from the file name
    match = re.match(r"(\w+)_(\d{8})_fundus_(\w+)_([LR]).png", file_name)
    patient_id, visit_date, fundus_id, laterality = match.groups()
    return patient_id, visit_date, fundus_id, laterality

def load_images(directory):
    # load all images in the directory
    image_data = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".png"):
            file_path = os.path.join(directory, file_name)
            image = cv2.imread(file_path, 0)
            patient_id, visit_date, fundus_id, laterality = extract_image_data(file_name)
            image_data.append((patient_id, visit_date, fundus_id, laterality, image))
    return image_data

def get_template_image(image_data, laterality):
    # get the template image for the given laterality (L or R)
    for _, _, _, lat, image in image_data:
        if lat == laterality:
            return image
    return None

def register_image(template_image, image):
    # perform non-rigid 2D registration on the given image using the template image
    # you can use any non-rigid 2D registration technique, such as free-form deformation (FFD) or Demons
    # for example, using the Demons registration method:
    reg_image = cv2.reg.apply_registration(template_image, image, "Demon")
    return reg_image

def register_images(image_data):
    # register all images
    registered_images = []
    for patient_id, visit_date, fundus_id, laterality, image in image_data:
        template_image = get_template_image(registered_images, laterality)
        if template_image is None:
            template_image = image
        reg_image = register_image(template_image, image)
        registered_images.append((patient_id, visit_date, fundus_id, laterality, reg_image))
    return registered_images

def save_images(registered_images, directory):
    # save the registered images to the specified directory
    for patient_id, visit_date, fundus_id, laterality, reg_image in registered_images:
        file_name = f"{patient_id}_{visit_date}_fundus_{fundus_id}_{laterality}.png"
        file_path = os.path.join(directory, file_name)
        cv2.imwrite(file_path, reg_image)

if __name__ == "__main__":
    # specify the input directory containing the fundus images
    input_directory = "fundus_images"
    # specify
