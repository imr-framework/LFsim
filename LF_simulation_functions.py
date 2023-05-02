################################################################################################################
# This file contains all the functions that are required for the simulation, segmentation and texture analysis #
################################################################################################################

import os
import random
import cv2
import numpy as np
import pydicom as pyd
import matplotlib
import matplotlib.pyplot as plt
import keaDataProcessing as keaProc
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
from skimage import data
from skimage.transform import resize
from skimage.filters import threshold_otsu
from scipy import ndimage
from skimage import morphology
from skimage.segmentation import flood_fill
from sklearn.utils import shuffle



## Function for reading dicom files
def read_dicoms(dirname):
    Nz = len(os.listdir(dirname))
    file_num = 0
    for files in os.listdir(dirname):
        if files.endswith('.dcm'):
            file_num = file_num + 1
            #print(os.path.join(dirname,files))
            ds = pyd.dcmread(os.path.join(dirname,files))
            Nx = ds.Rows
            Ny = ds.Columns
            if (file_num == 1):
                # mat = ds.pixel_array
                # mat = np.reshape(mat,(Nx,Ny,1))
                Im_HF = np.zeros((Nx,Ny,Nz))
                Im_HF[:,:,ds.InstanceNumber-1] = ds.pixel_array
            else:
                # temp = np.reshape(ds.pixel_array, (Nx, Ny, 1))
                # mat = np.append(mat, temp, axis=2)
                Im_HF[:, :, ds.InstanceNumber - 1] = ds.pixel_array

    return Im_HF, ds, Nz


## Function for reading dicom files of IMA type
def read_dicoms_IMA(dirname):
    Nz = len(os.listdir(dirname))
    file_num = 0
    for files in os.listdir(dirname):
        if files.endswith('.IMA'):
            file_num = file_num + 1
            #print(os.path.join(dirname,files))
            ds = pyd.read_file(os.path.join(dirname,files))
            Nx = ds.Rows
            Ny = ds.Columns
            if (file_num == 1):
                # mat = ds.pixel_array
                # mat = np.reshape(mat,(Nx,Ny,1))
                Im_HF = np.zeros((Nx,Ny,Nz))
                Im_HF[:,:,ds.InstanceNumber-1] = ds.pixel_array
            else:
                # temp = np.reshape(ds.pixel_array, (Nx, Ny, 1))
                # mat = np.append(mat, temp, axis=2)
                Im_HF[:, :, ds.InstanceNumber - 1] = ds.pixel_array

    return Im_HF, ds, Nz


## Function for reading NIFTI files
def read_nifti(dirname):
    nifti_data = nib.load(dirname)
    Im_HF      = nifti_data.get_fdata()
    hdr        = nifti_data.header

    return Im_HF, hdr


## Function for saving as .npy file
def save_numpy(dirname, variable):
    # complete_path = os.path.join(dirname, filename)
    np.save(dirname, variable)  # dirname should have the name that has to be given to the saved file

    return


## Function for Normalization
def normalize(data):
    x_max = np.max(data)
    x_min = np.min(data)

    norm_data = (data - x_min)/(x_max - x_min)

    return norm_data


## Function for calculating Contrast map
def Contrast_map(TR, T1L, T1H, Im_HF_tissue, noise_std = 0.1):

    T1L_noise = np.zeros((Im_HF_tissue.shape))
    T1L_noise = T1L_noise + T1L + noise_std*np.random.randn(T1L_noise.shape[0], T1L_noise.shape[1], T1L_noise.shape[2])       # 0.05 is the SD in T1
    T1H_noise = np.zeros((Im_HF_tissue.shape))
    T1H_noise = T1H_noise + T1H + noise_std*np.random.randn(T1H_noise.shape[0], T1H_noise.shape[1], T1H_noise.shape[2])       # 0.05 is the SD in T1
    a = (1 - np.exp(-TR/T1L_noise))
    b = (1 - np.exp(-TR/T1H_noise))
    c = Im_HF_tissue
    signal_LF = (a/b)*c

    return signal_LF


## Function for Down Res
def resize_data(data, res, matrix, new_res_x=1.5, new_res_y=1.5, new_res_z=1.5):
    # initial_size_x = data.shape[0]                           # extract x,y,z info from image data
    # initial_size_y = data.shape[1]                           # This information is matrix size
    # initial_size_z = data.shape[2]

    new_size_x = int(np.round(res[0]*matrix[0])/new_res_x)   # define new dimensions of matrix size
    new_size_y = int(np.round(res[1]*matrix[1])/new_res_y)   # (Resolution * matrix size) / 1.5 = New matrix size
    new_size_z = int(np.round(res[2]*matrix[2])/new_res_z)   # 1.5 is the new resolution that we want

    # delta_x = initial_size_x / new_size_x                    # ratio of old to new matrix size
    # delta_y = initial_size_y / new_size_y                    # (or)
    # delta_z = initial_size_z / new_size_z                    # ratio of new to old resolution

    image_resized = resize(data, (new_size_x, new_size_y, new_size_z))

    return image_resized


## Function for making noise matrix from the VLF scans (10 scans)
def compile_noise(dirname, dirname_2):

    # giving file extension
    ext = ('.3d')

    # initial patch
    image_path = dirname_2
    kSpace = keaProc.readKSpace(image_path)
    LF_acq = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(kSpace)))

    I1 = np.abs(LF_acq)
    I1 = normalize(I1)
    patch = 16
    finalArray1 = I1[:, 0:patch, 0:patch]
    finalArray2 = I1[:, -patch:, 0:patch]

    # iterating over all files
    for files in os.listdir(dirname):
        if files.endswith(ext):
            os.chdir(dirname)
            kSpace = keaProc.readKSpace(files)
            LF_acq = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(kSpace)))

            I = np.abs(LF_acq)
            I = normalize(I)
            p1 = I[:, 0:patch, 0:patch]
            p2 = I[:, -patch:, 0:patch]
            finalArray1 = np.concatenate([finalArray1, p1], axis=1)
            finalArray2 = np.concatenate([finalArray2, p2], axis=2)
        else:
            print('done')

    finalArray1 = np.swapaxes(finalArray1, 1, 2)

    for files in os.listdir(dirname):
        finalArray1 = np.concatenate([finalArray1, finalArray2], axis=1)
    LF_noise = finalArray1

    return LF_noise


## Function for cutting out noise patch from noise matrix to add to the image - this does not add noise
def get_noise(data, LF_noise, noise_fact):
    x = data.shape[0]
    y = data.shape[1]
    z = data.shape[2]

    l = LF_noise.shape[0]
    m = LF_noise.shape[1]
    n = LF_noise.shape[2]

    LF_noise = np.ndarray.flatten(LF_noise)

    p = np.random.permutation(l*m*n)

    LF_noise_use = LF_noise[p[0:x*y*z]]
    LF_noise_use = np.reshape(LF_noise_use, (x,y,z))
    temp1 = LF_noise_use * noise_fact

    return temp1


## Function for Noise estimation in High Field
def compute_noise_HF(data):
    patch_size = 4
    # noise_sample1 = data[:, 0:patch_size, 0:patch_size]
    # noise_sample2 = data[:, -patch_size:, 0:patch_size]

    noise_sample1 = data[0:patch_size, 0:patch_size, :]
    noise_sample2 = data[-patch_size:, 0:patch_size, :]

    noise_sample_combined = np.concatenate([noise_sample1,noise_sample2], axis=0)
    std = np.std(noise_sample_combined)
    # OrthoSlicer3D(noise_sample1).show()

    return std


## Function for Noise estimation in Low Field acquired
def compute_noise_LF(data):
    patch_size = 16
    noise_sample1 = data[:, 0:patch_size, 0:patch_size]
    noise_sample2 = data[:, -patch_size:, 0:patch_size]
    noise_sample_combined = np.concatenate([noise_sample1,noise_sample2], axis=0)
    std = np.std(noise_sample_combined)
    # OrthoSlicer3D(noise_sample2).show()

    return std


## Function for calculating Signal
def compute_signal(data, LF_sim_mask=0, thresh = 0.2):
    if np.sum(LF_sim_mask) == 0:
        # thresh = threshold_otsu(data)
        tissue_mask = data > thresh
        tissue_mask = ndimage.binary_erosion(tissue_mask)    # this is scipy library
        LF_sim_mask = ndimage.binary_dilation(tissue_mask)

    # tissue_mask = morphology.binary_erosion(tissue_mask)   # this is skimage library
    # tissue_mask = morphology.binary_dilation(tissue_mask)
    tissue = data * LF_sim_mask
    tissue_size = np.count_nonzero(tissue)
    signal = np.sum(tissue)/tissue_size
    signal_max = np.max(tissue)

    return signal, signal_max, LF_sim_mask


## Function for calculating SNR
def compute_snr(mu, sigma):
    snr = mu/sigma

    return snr


## Experiment function
def get_sigma_n(mu_HF, mu_LT_practical, sigma_HF, sigma_LT_practical):
    sigma_n = ((mu_HF * sigma_LT_practical) - (mu_LT_practical * sigma_HF)) / mu_LT_practical
    print('sigma_n: ', sigma_n)

    return sigma_n


## Function for extracting slice
def extract_slice(data, slice_number, fcm=0):
    if fcm == 0:
        slice = data[:, :, slice_number]

    elif fcm == 1:
        slice_number_fcm = int((slice_number*120)/30)
        # because T2 FLAIR has 30 slices whereas FCM has 120 slices. So extracting the corresponding slice
        slice = data[:, :, slice_number_fcm]

    return slice


## Function for removing background noise
def remove_background_noise(slice, slice_fcm):

    slice_fcm[slice_fcm == 1] = 0  # background
    slice_fcm[slice_fcm == 2] = 1  # CSF
    slice_fcm[slice_fcm == 3] = 1  # Grey Matter
    slice_fcm[slice_fcm == 4] = 1  # White Matter
    # converting all tissue labels to 1 because we need to remove seperate background from brain. So all brain tissues are labeled as 1 and background as 0
    # this step is only for HF data. For LF, we are using manual mask, so its already binary.

    brain_tissue = slice * slice_fcm

    return brain_tissue


## Function for removing background noise in LF images
def remove_background_noise_LF(slice, slice_fcm):

    # converting all tissue labels to 1 because we need to remove seperate background from brain. So all brain tissues are labeled as 1 and background as 0
    # this step is only for HF data. For LF, we are using manual mask, so its already binary.
    brain_tissue = slice * slice_fcm

    return brain_tissue


## Function for extracting lesion by thresholding
def make_lesion_mask(lesion_mask, thresh=250, nhp=8078):

    if nhp == 8078:
        lesion_mask = lesion_mask > thresh
        lesion_mask = lesion_mask.astype(int)
        # normal thresholding

    elif nhp == 8112 or nhp == 8671:
        lesion_mask = lesion_mask > thresh
        lesion_mask = lesion_mask.astype(int)
        lesion_mask = ndimage.binary_erosion(lesion_mask)
        # erosion after threshold is required to remove extra clutter

    # # to remove extra bits
    # lesion_mask = ndimage.binary_erosion(lesion_mask)
    # lesion_mask = ndimage.binary_dilation(lesion_mask)

    # lesion = data * lesion_mask

    return lesion_mask


## Function for extracting lesion by thresholding
def make_lesion_mask_LF(lesion_mask, thresh=0.32, nhp=8078):

    if nhp == 8078:
        thresh = 0.32
        lesion_mask = lesion_mask > thresh
        lesion_mask = lesion_mask.astype(int)
        lesion_mask = ndimage.binary_erosion(lesion_mask, structure=np.ones((2,2)))     # to remove extra bits
        # normal thresholding

    elif nhp == 8112:
        thresh = 0.4
        lesion_mask = lesion_mask > thresh
        lesion_mask = lesion_mask.astype(int)
        # lesion_mask = ndimage.binary_erosion(lesion_mask, structure=np.ones((2, 2)))

    elif nhp == 8671:
        thresh = 0.4
        lesion_mask = lesion_mask > thresh
        lesion_mask = lesion_mask.astype(int)
        # lesion_mask = ndimage.binary_erosion(lesion_mask, structure=np.ones((2, 2)))  #structure=np.ones((2, 2))
        lesion_mask = ndimage.binary_dilation(lesion_mask)
        # erosion after threshold is required to remove extra clutter

    return lesion_mask


## Function for finding contours and their centroid
def find_centroid(lesion_mask):
    # finding contour
    lesion_mask = lesion_mask.astype('uint8')
    contours, hierarchy = cv2.findContours(lesion_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # finding centroid
    list_cX = []
    list_cY = []
    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)

        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        list_cX.append(cX)
        list_cY.append(cY)

    array_cX = np.array(list_cX)
    array_cY = np.array(list_cY)

    return array_cX, array_cY


## Function for finding contours and their centroid
def find_centroid_LF(lesion_mask, nhp=8671):
    if nhp == 8078 or nhp == 8112:
        # Finding white pixels
        all_coord = np.nonzero(lesion_mask)
        lesion_coord = [all_coord[0][-1], all_coord[1][-1]]
        # lesion segmentation is giving multiple pixels with value 1. One of them belongs to lesion.
        # Here we are extracting last coordinate as that belongs to lesion in all NHPs.
        cx, cy = lesion_coord[0], lesion_coord[1]

    if nhp == 8671:
        # finding contour
        lesion_mask = lesion_mask.astype('uint8')
        contours, hierarchy = cv2.findContours(lesion_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # finding centroid
        list_cX = []
        list_cY = []
        for c in contours:
            # calculate moments for each contour
            M = cv2.moments(c)

            # calculate x,y coordinate of center
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            list_cX.append(cX)
            list_cY.append(cY)

        cx = np.array(list_cX)
        cy = np.array(list_cY)

    return cx, cy


## Function for region growing - for testing - not being used now
def region_growing_for_testing(slice, cx, cy, nhp = 8078):

    lesion_mask_final = np.zeros(slice.shape)

    # for i in range(len(cx)):
    lesion_temp = flood_fill(slice, (cy[1], cx[1]), 1000, tolerance=20)
    lesion_mask_temp = lesion_temp > 999
        # lesion_mask_final = lesion_mask_final + lesion_mask_temp

    # lesion_mask_final = 1 - lesion_mask_final
    lesion = slice * lesion_mask_temp

    return lesion, lesion_mask_final


## Function for region growing
def region_growing(slice, cx, cy, nhp = 8671):

    lesion_mask_final = np.zeros(slice.shape)

    if nhp == 8078 or nhp == 8112:
        for i in range(len(cx)):
            lesion_temp = flood_fill(slice, (cy[i], cx[i]), 1000, tolerance=20)
            # apply region growing on input image and assign the grown area the pixel intensity of 1000

            lesion_mask_temp = lesion_temp > 999
            # threshold to get the RG output separately as a mask

            lesion_mask_final = lesion_mask_final + lesion_mask_temp
            # add this mask to array of zeros to get RG output mask

        # lesion_mask_final = 1 - lesion_mask_final
        lesion = slice * lesion_mask_final
        # multiplication with input to get segmentation

    if nhp == 8671:
        cx_new = cx[cx != 0]
        cy_new = cy[cy != 0]
        # the function for finding centroid is finding multiple centroids because other brain areas are coming up as well.
        # these multiple centroids have cordinates (0,0) as well (don't know why!). Therefore, removing these (0,0) coordinates.

        lesion_temp = flood_fill(slice, (cy_new[0], cx_new[0]), 1000, tolerance=20)
        # in this NHP, other areas are also getting segmented. Therefore, choose only the first centroid which belongs to the lesion and then apply RG

        # the rest of the steps are same as other NHPs
        lesion_mask_temp = lesion_temp > 999
        lesion_mask_final = lesion_mask_final + lesion_mask_temp
        lesion = slice * lesion_mask_final

    return lesion, lesion_mask_final


## Function for region growing
def region_growing_LF(slice, cx, cy, nhp = 8078):

    lesion_mask_final = np.zeros(slice.shape)

    if nhp == 8078 or nhp == 8112:
        lesion_temp = flood_fill(slice, (cx, cy), 1000, tolerance=0.05)
        # apply region growing on input image and assign the grown area the pixel intensity of 1000

        lesion_mask_temp = lesion_temp > 999
        # threshold to get the RG output separately as a mask

        lesion_mask_final = lesion_mask_final + lesion_mask_temp
        # add this mask to array of zeros to get RG output mask

        # lesion_mask_final = 1 - lesion_mask_final
        lesion = slice * lesion_mask_final
        # multiplication with input to get segmentation

    if nhp == 8671:
        lesion_temp = flood_fill(slice, (cx, cy), 1000, tolerance=0.02)
        # in this NHP, other areas are also getting segmented. Therefore, choose only the first centroid which belongs to the lesion and then apply RG

        # the rest of the steps are same as other NHPs
        lesion_mask_temp = lesion_temp > 999
        lesion_mask_final = lesion_mask_final + lesion_mask_temp
        lesion = slice * lesion_mask_final

    return lesion, lesion_mask_final


## Function for making patches for DL training - with shuffling of pixels within the patch - not useful - but lets keep it
def make_patches_with_shuffle(slice_mask, number_of_patches, size_of_patches):
    final_patch = list()

    if size_of_patches > slice_mask.shape[0]:
        print('\nPatch size is larger than image size. Please enter smaller size.')
    else:
        for i in range(0,number_of_patches):
            l = slice_mask.shape[0]
            m = slice_mask.shape[1]

            x = size_of_patches
            y = size_of_patches

            slice_mask_flat = np.ndarray.flatten(slice_mask)

            p = np.random.permutation(l * m)

            p1 = slice_mask_flat[p[0:x * y]]
            p1 = np.reshape(p1, (x, y))
            final_patch.append(p1)

        final_patch = np.array(final_patch)
        final_patch = np.swapaxes(final_patch, 0, 2)

    return final_patch


## Function for making patches and labelling them as 0 or 1 for DL training - it is class imbalanced
def data_augmentation_class_imbalanced(slice, slice_mask, number_of_patches, size_of_patches):

    x_train = np.zeros((size_of_patches, size_of_patches, number_of_patches))
    y_train = np.zeros((size_of_patches, size_of_patches, number_of_patches))
    y_label = np.zeros(number_of_patches)

    if size_of_patches > slice_mask.shape[0]:
        print('\nPatch size is larger than image size. Please enter smaller size.')
    else:
        for i in range(0,number_of_patches):
            l = slice_mask.shape[0]
            m = slice_mask.shape[1]

            p_l = size_of_patches
            p_m = size_of_patches

            c_x = random.randint(0, l - p_l - 1)
            c_y = random.randint(0, m - p_m - 1)

            x_patch = slice[int(c_x):int(c_x + p_l), int(c_y):int(c_y + p_m)]
            y_patch = slice_mask[int(c_x):int(c_x + p_l), int(c_y):int(c_y + p_m)]

            x_train[:, :, i] = x_patch
            y_train[:, :, i] = y_patch
            y_train = y_train.astype('int')

            if (np.sum(y_patch) > 0):
                y_label[i] = 1
            else:
                y_label[i] = 0

        y_label = y_label.astype('int')

    return x_train, y_train, y_label


## Function for making patches and labelling them as 0 or 1 for DL training - class balanced
# this function's algorithm was working fine for small number of patches. For larger number, it failed and therefore not using it anymore.
def data_augmentation_inefficient(slice, slice_mask, number_of_patches, size_of_patches):

    x_train = np.zeros((size_of_patches, size_of_patches, number_of_patches))
    y_train = np.zeros((size_of_patches, size_of_patches, number_of_patches))
    y_label = np.zeros(number_of_patches)

    l = slice_mask.shape[0]
    m = slice_mask.shape[1]

    p_l = size_of_patches
    p_m = size_of_patches

    if size_of_patches > slice_mask.shape[0]:
        print('\nPatch size is larger than image size. Please enter smaller size.')
    elif (number_of_patches % 2) != 0:
        print('\nNumber of patches is an odd number. Please enter Even number!')
    else:
        for i in range(0,number_of_patches):
            c_x = random.randint(0, l - p_l - 1)
            c_y = random.randint(0, m - p_m - 1)

            x_patch = slice[int(c_x):int(c_x + p_l), int(c_y):int(c_y + p_m)]
            y_patch = slice_mask[int(c_x):int(c_x + p_l), int(c_y):int(c_y + p_m)]

            x_train[:, :, i] = x_patch
            y_train[:, :, i] = y_patch
            y_train = y_train.astype('int')

            if (np.sum(y_patch) > 0):
                y_label[i] = 1
            else:
                y_label[i] = 0

        y_label = y_label.astype('int')

        for i in y_label:
            if y_label[i] == 0:
                while True:
                    c_x = random.randint(0, l - p_l - 1)
                    c_y = random.randint(0, m - p_m - 1)

                    x_patch = slice[int(c_x):int(c_x + p_l), int(c_y):int(c_y + p_m)]
                    y_patch = slice_mask[int(c_x):int(c_x + p_l), int(c_y):int(c_y + p_m)]

                    if (np.sum(y_patch) > 0):
                        x_train[:, :, i] = x_patch
                        y_train[:, :, i] = y_patch
                        y_label[i] = 1
                        i = i + 1
                        number_of_label_1 = np.count_nonzero(y_label == 1)
                        if number_of_label_1 == number_of_patches/2:
                            break

    x_train = np.swapaxes(x_train, 0, 2)
    y_train = np.swapaxes(y_train, 0, 2)
    x_train, y_train, y_label = shuffle(x_train, y_train, y_label, random_state=0)

    return x_train, y_train, y_label


## Function for making patches and labelling them as 0 or 1 for DL training - class balanced - Efficient method
def data_augmentation(slice, slice_mask, number_of_patches, size_of_patches):

    x_train_0 = np.zeros((size_of_patches, size_of_patches, int(number_of_patches/2)))
    y_train_0 = np.zeros((size_of_patches, size_of_patches, int(number_of_patches/2)))
    y_label_0 = int(number_of_patches/2)

    x_train_1 = np.zeros((size_of_patches, size_of_patches, int(number_of_patches/2)))
    y_train_1 = np.zeros((size_of_patches, size_of_patches, int(number_of_patches/2)))
    y_label_1 = int(number_of_patches/2)

    l = slice_mask.shape[0]
    m = slice_mask.shape[1]

    p_l = size_of_patches
    p_m = size_of_patches

    i = 0
    j = 0

    if size_of_patches > slice_mask.shape[0]:
        print('\nPatch size is larger than image size. Please enter smaller size.')
    elif (number_of_patches % 2) != 0:
        print('\nNumber of patches is an odd number. Please enter Even number!')
    else:

        while y_label_1 > 0:
            c_x = random.randint(0, l - p_l - 1)
            c_y = random.randint(0, m - p_m - 1)

            x_patch_1 = slice[int(c_x):int(c_x + p_l), int(c_y):int(c_y + p_m)]
            y_patch_1 = slice_mask[int(c_x):int(c_x + p_l), int(c_y):int(c_y + p_m)]

            if (np.sum(y_patch_1) > 0):
                y_label_1 = y_label_1 - 1
                x_train_1[:, :, i] = x_patch_1
                y_train_1[:, :, i] = y_patch_1
                i = i + 1

        y_label_1_array = np.ones(int(number_of_patches/2))

        while y_label_0 > 0:
            c_x = random.randint(0, l - p_l - 1)
            c_y = random.randint(0, m - p_m - 1)

            x_patch_0 = slice[int(c_x):int(c_x + p_l), int(c_y):int(c_y + p_m)]
            y_patch_0 = slice_mask[int(c_x):int(c_x + p_l), int(c_y):int(c_y + p_m)]

            if (np.sum(y_patch_0) == 0):
                y_label_0 = y_label_0 - 1
                x_train_0[:, :, j] = x_patch_0
                y_train_0[:, :, j] = y_patch_0
                j = j + 1

        y_label_0_array = np.zeros(int(number_of_patches/2))

        x_train_0 = np.concatenate([x_train_0, x_train_1], axis=2)
        y_train_0 = np.concatenate([y_train_0, y_train_1], axis=2)
        y_label_0_array = np.concatenate([y_label_0_array, y_label_1_array])

        x_train = x_train_0
        y_train = y_train_0
        y_label = y_label_0_array

        x_train = x_train.astype('int')
        y_train = y_train.astype('int')
        y_label = y_label.astype('int')

        x_train = np.swapaxes(x_train, 0, 2)        # swaping axes because shuffle needs all axis=0 to be same. y_label has different axis=0 as compared to x_train and y_train
        y_train = np.swapaxes(y_train, 0, 2)
        x_train, y_train, y_label = shuffle(x_train, y_train, y_label, random_state=0)

    return x_train, y_train, y_label


## Function for converting NHP input dictionary into 3D array of patches
def convert_input2array(data, number_of_patches, size_of_patches):
    # extract x_train and y_train keys from dictionary
    x_train = data.item().get('x_train')
    y_train = data.item().get('y_train')

    # Create patches from first slice to initialize further concatenation
    slice_x_init = x_train[0]
    slice_y_init = y_train[0]
    # slice_x_init = 4096 * slice_x_init    # increasing dynamic range here before cutting patches. To be done within function for LF data.
    # slice_y_init = 4096 * slice_y_init    # increasing dynamic range here before cutting patches - not required
    slice_x_init_patch, slice_y_init_patch, slice_y_init_label = data_augmentation(slice_x_init, slice_y_init,
                                                                                   number_of_patches, size_of_patches)

    # create patches from remaining slices and combine them all together
    for i in range(len(x_train) - 1):
        slice_x = x_train[i + 1]
        slice_y = y_train[i + 1]
        # slice_x = 4096 * slice_x    # increasing dynamic range here before cutting patches. To be done within function for LF data.
        # slice_y = 4096 * slice_y    # increasing dynamic range here before cutting patches - not required
        print('Slice number being patchified: ', i + 1)

        slice_x_patch, slice_y_patch, slice_y_label = data_augmentation(slice_x, slice_y, number_of_patches,
                                                                        size_of_patches)

        slice_x_init_patch = np.concatenate([slice_x_init_patch, slice_x_patch], axis=0)
        slice_y_init_patch = np.concatenate([slice_y_init_patch, slice_y_patch], axis=0)

    x_train_final = slice_x_init_patch
    y_train_final = slice_y_init_patch

    return x_train_final, y_train_final
