#Importing all the necessary modules
print("Importing modules")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os #File operations
import cv2
import math
from tqdm import tqdm
from PIL import Image
print("Modules imported")



base_dir = 'D:/osic-pulmonary-fibrosis-progression/'
base_dir_save = 'D:/osic-pulmonary-fibrosis-progression/processed/'
train_dir = base_dir + 'train/'
test_dir = base_dir + 'test/'
processed_train_dir = base_dir_save + 'train/'
processed_test_dir = base_dir_save + 'test/'


#Get all train patient ids from directories
train_patient_ids=[]
for root,dirs,filenames in os.walk(train_dir):
    train_patient_ids.extend(dirs)
print(len(train_patient_ids))

#Get all test patient ids from directories
test_patient_ids=[]
for root,dirs,filenames in os.walk(test_dir):
    test_patient_ids.extend(dirs)
print(len(test_patient_ids))


#For removing borders on dicom images
def crop_image(img: np.ndarray):
    if img.shape[0] != img.shape[1]:
        edge_pixel_value = img[0, 0]
        mask = img != edge_pixel_value
        return img[np.ix_(mask.any(1),mask.any(0))]
    return img

NUM_SLICES = 30         #Fixed number of slices
IMG_PX_SIZE = 224

#Progressively returns chunks
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def mean(l):
    return sum(l)/len(l)


#Making all the slices of same size
def resize_slices(slices):
    slices = [cv2.resize(slice , (IMG_PX_SIZE , IMG_PX_SIZE)) for slice in slices]
    if len(slices) == NUM_SLICES:
        return slices
    else:
        chunk_size = int(np.ceil(len(slices) / NUM_SLICES))
        new_slices = []
        for chunk in chunks(slices , chunk_size):
            chunk = list(map(mean , zip(*chunk)))
            new_slices.append(chunk)
        if len(new_slices) < NUM_SLICES:
            for i in range(NUM_SLICES - len(new_slices)):
                new_slices.append(new_slices[-1])
        elif len(new_slices) > NUM_SLICES:
            extra = new_slices[NUM_SLICES-1:]
            last = list(map(mean , zip(*extra)))
            del new_slices[NUM_SLICES:]
            new_slices[-1] = last
        return new_slices

BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618'] #Bad Encoding

for patient_id in tqdm(train_patient_ids):
    if patient_id in BAD_ID:
        continue
    path = train_dir + patient_id
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    #Adding metadata Slice Thickness to all the slices 
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = slices[0].SliceThickness
    for s in slices:
        s.SliceThickness = slice_thickness
    
    #Removing borders on all slices
    cropped_slices = [crop_image(np.array(slice.pixel_array)) for slice in slices]
    #Resizing slices
    processed_slices = resize_slices(cropped_slices)
    
    #For saving the dicom files as jpeg images (Not necessary to do)
    for num,slice in enumerate(processed_slices,1):
        path_to_save = processed_train_dir + patient_id + '/' + str(num) + '.jpeg'
        if not os.path.exists(os.path.dirname(path_to_save)):
            try:
                os.makedirs(os.path.dirname(path_to_save))
            except OSError as exc:
                print(exc)  
        slice = np.array(slice)
        #Normalization for saving dicom files as jpeg
        norm = (slice.astype(np.float)-slice.min())*255.0 / (slice.max()-slice.min()) 
        Image.fromarray(norm.astype(np.uint8)).save(path_to_save)