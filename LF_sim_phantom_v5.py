#######################################################
## Low Field Simulation of High Field Phantom images ##
#######################################################

import numpy as np
import pydicom as pyd
import matplotlib
import matplotlib.pyplot as plt
import keaDataProcessing as keaProc
from pyfeats import glcm_features
from roipoly import RoiPoly
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
from LF_simulation_functions import read_dicoms
from LF_simulation_functions import read_dicoms_IMA
from LF_simulation_functions import read_nifti
from LF_simulation_functions import normalize
from LF_simulation_functions import Contrast_map
from LF_simulation_functions import resize_data
from LF_simulation_functions import compile_noise
from LF_simulation_functions import get_noise
from LF_simulation_functions import compute_noise_HF
from LF_simulation_functions import compute_noise_LF
from LF_simulation_functions import compute_signal
from LF_simulation_functions import compute_snr
from LF_simulation_functions import get_sigma_n

##############
# User input #
##############

## Change these numbers according to the neuroanatomy
nifti_input     = False         # make it true if you want to use NIFTI data
dicom_input     = False         # make it true if you want to use DICOM data - for NHP
IMA_dicom_input = True          # make it true if you want to use DICOM IMA data - for phantom

contrast_component   = False    # make it true if you want to add contrast component
resolution_component = True     # make it true if you want to add resolution component
snr_component        = True     # make it true if you want to add SNR component

viewing = True                  # make it true to see output
# Sequence of output - (1)HF, (2)HF resized, (3)LF simulated, (4)LF acquired

# New Resolution - change these to get desired resolution
new_res_x = 1.48
new_res_y = 1.48
new_res_z = 5.0

# Threshold for segmenting phantom - don't change until required. It affects SNR component.
thresh = 0.2

## All paths
image_path_HF   = r"C:\Mount_Sinai\5_phantom_data\High_Field\T2_FLAIR_0010_for_LF_sim"
image_path_LF   = r"C:\Mount_Sinai\6_data_for_noise\data_axial\data1.3d"
acqu_path       = r"C:\Mount_Sinai\6_data_for_noise\acqu_axial\acqu1.par"
noise_path      = r"C:\Mount_Sinai\6_data_for_noise\data_axial"
B0_corr_path    = r"C:\Mount_Sinai\5_phantom_data\Low_Field\B0_corrected\7_TSE_after_correction_filt.npy"

# Note:
# image_path_HF - path to 5_phantom_data folder
# image_path_LF - any data.3d file out of the 6_data_for_noise folder
# acqu_path - corresponding acqu.par file out of the 6_data_for_noise folder
# noise_path - path to the repeatability dataset for making noise matrix which will be used for noise simulation (6_data_for_noise)


############################
# Initialization of values #
############################

## Values according to Andrew Webb's paper - Low Field MRI: An MR Physics Perspective (J. Magn. Reson. Imaging 2019)
# SNR is proportional to 3/2 power of B0
# T1 (ms) = a(gamma_bar.B0)^b
# where gamma_bar is the gyromagnetic ratio
gamma_bar = 42.57 * (10**6)

# a and b are constants that different values according to the tissue - for CSF these values are stable
a_WM = 0.71
a_GM = 1.16
b_WM = 0.382
b_GM = 0.376

# High field strength is 3T and Low Field strength is 50 mT
B0_HF = 3
B0_LF = 0.05

# TR value in milliseconds
TR = 500


## T1 constants from literature

# Low Field - Tom O'Reilly and Andrew Webb, MRM, 2021
T1_GM_LF_mean = 335
T1_GM_LF_std  = 25

T1_WM_LF_mean = 280
T1_WM_LF_std  = 15

# Low Field - Wasanapura, JMRI, 1999
T1_GM_HF_mean = 1300
T1_GM_HF_std  = 50

T1_WM_HF_mean = 832
T1_WM_HF_std  = 20


##################
# Importing data #
##################

## Import High field image - DICOM
if dicom_input == True:
    Im_HF, ds, Nz_HF = read_dicoms(image_path_HF)                 # this function returns 3 variables - image (Im_HF), header information (ds) and number of slices (Nz_HF)
    Im_HF = Im_HF.astype(float)                                   # calculations on integers causes problem. Therefore converting the image to float
    print('Matrix size of High Field image: ', Im_HF.shape)


## Import High Field image - DICOM IMA
if IMA_dicom_input == True:
    Im_HF, ds, Nz_HF = read_dicoms_IMA(image_path_HF)             # this function returns 3 variables - image (Im_HF), header information (ds) and number of slices (Nz_HF)
    Im_HF = Im_HF.astype(float)                                   # calculations on integers causes problem. Therefore converting the image to float
    print('Matrix size of High Field image: ', Im_HF.shape)


## Import High Field image - NIFTI
if nifti_input == True:
    Im_HF, hdr = read_nifti(image_path_HF)                        # this function read nifti file. It returns 2 variables - image (Im_HF) and header info (hdr)
    print('Matrix size of High Field image: ', Im_HF.shape)


## Import acquired Low Field image
ImageScanParams = keaProc.readPar(acqu_path)                             # this function reads acqu.par file coming from the low field scanner
kSpace          = keaProc.readKSpace(image_path_LF)                      # this function reads data.3d file which is the k-space data from scanner
LF_acq          = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(kSpace)))  # fast fourier transform of k-space to form the image
LF_acq          = np.abs(LF_acq)                                         # converting the complex values of image to absolute values
fov_LF_acq      = ImageScanParams.get('FOV')
matrix_LF_acq   = LF_acq.shape
res_LF_acq      = np.divide(fov_LF_acq,matrix_LF_acq)
print('Matrix size of acquired Low Field image: ', matrix_LF_acq)
print('FOV of acquired LF: ', fov_LF_acq)
print('Resolution of acquired LF: ', res_LF_acq)
# LF acquired image parameters:
# FOV - 230 230 125
# Matrix size - 155 155 25
# Resolution - 1.48 1.48 5


## Import B0 corrected acquired LF
B0_corr = np.load(B0_corr_path)
B0_corr = np.abs(B0_corr)


## Normalize High field image to 0 and 1
Im_HF = normalize(Im_HF)


############
# Contrast #
############
# We still need to find a way to validate the contrast in phantom. For now, we are NOT considering this part because phantom is made of one material only

if contrast_component == True:
    ## Read HF segmentation masks: 1 - background, 2 - CSF, 3 - GM, 4 - WM
    mask, mask_header, Nz_mask = read_dicoms(mask_path)

    ## Mask tissue wise
    mask_CSF = np.zeros(mask.shape)
    mask_CSF[mask==2] = 1
    Im_LF_CSF = np.multiply(Im_HF, mask_CSF)   #No change in T1 value across field strengths therefore HF and LF masks are same for CSF
    # OrthoSlicer3D(seg_CSF).show()

    mask_GM = np.zeros(mask.shape)
    mask_GM[mask==3] = 1
    Im_HF_GM = np.multiply(Im_HF, mask_GM)     # HF GM contrast map
    # OrthoSlicer3D(mask_GM).show()

    mask_WM = np.zeros(mask.shape)
    mask_WM[mask==4] = 1
    Im_HF_WM = np.multiply(Im_HF, mask_WM)     # HF WM contrast map

    Im_LF_GM = Contrast_map(TR, T1_GM_LF_mean, T1_GM_HF_mean, Im_HF_GM)    # LF GM contrast map using function because we don't have LF masks

    Im_LF_WM = Contrast_map(TR, T1_WM_LF_mean, T1_WM_HF_mean, Im_HF_WM)    # LF WM contrast map using function because we don't have LF masks

    # Final LF contrasts combined
    Im_LF_contrast = Im_LF_WM + Im_LF_GM + Im_LF_CSF
    print('Shape of LF contrasts combined: ', Im_LF_contrast.shape)


##########################
# Down Res of High field #
##########################

if resolution_component == True:
    if 'Im_LF_contrast' in globals():
        input_for_down_res = Im_LF_contrast
        print('\n Contrast component detected and used')
        print('\n Down Resolution started')
    else:
        input_for_down_res = Im_HF
        print('\n Contrast component not detected')
        print('\n Down Resolution started')

    ## for NIFTI input
    if nifti_input == True:
        # hdr    = Im_HF.header                                        # reading header information
        dim     = hdr['dim']                                          # extracting acquisition matrix size - it has some other useless values as well
        pixdim  = hdr['pixdim']                                       # extracting resolution - it has some other useless values as well
        matrix  = [dim[1], dim[2], dim[3]]                            # taking only relevant values from matrix size
        res     = [pixdim[1], pixdim[2], pixdim[3]]                   # taking only relevant values from resolution
        Im_LFS_contrast_resize  = resize_data(Im_HF,res,matrix)      # resizing image

        print('Original High Field Matrix size:', matrix[0], matrix[1], matrix[2])
        print('Original High Field Resolution:', res[0], res[1], res[2])
        print('Matrix Size after Down Res: ', Im_LFS_contrast_resize.shape)


    ## for DICOM and IMA DICOM input
    if dicom_input or IMA_dicom_input == True:
        matrix                 = [np.max(ds.AcquisitionMatrix), np.max(ds.AcquisitionMatrix), Nz_HF]            # matrix creation from header information
        res                    = [ds.PixelSpacing[0], ds.PixelSpacing[1], ds.SliceThickness]      # Nz_HF/Nz_HF]                          # resolution extraction from header information
        Im_LFS_contrast_resize = resize_data(input_for_down_res, res, matrix, new_res_x, new_res_y, new_res_z)    # image resizing

        print('Original High Field Matrix size:', matrix[0], matrix[1], matrix[2])
        print('Original High Field Resolution:', res[0], res[1], res[2])
        print('Matrix Size after Down Res: ', Im_LFS_contrast_resize.shape)

    print('Down Resolution finished')


###################
####### SNR #######
###################

if snr_component == True:
    if 'Im_LFS_contrast_resize' in globals():
        input_for_snr = Im_LFS_contrast_resize
        print('\n Down res component detected and used')
        print('\n SNR component started')
    else:
        input_for_snr = Im_HF
        print('\n Down res component not detected')
        print('\n SNR component started')

    ## Normalization
    Im_LFS_contrast_resize  = normalize(input_for_snr)                      # Down res output will be normalized to get into SNR code
    LF_acq                  = normalize(LF_acq)                             # acquired low field will also be normalized to get into SNR code
    B0_corr                 = normalize(B0_corr)


    ## SNR calculation in acquired Low Field
    weight = 1
    # This weight factor was included because initially we were just adding noise without considering SNR.
    # So to bring that simulation at the level of acquired LF, we had to include a weight factor for noise addition. Now it is not useful. Hence value 1.
    sigma_LF_acq                             = compute_noise_LF(LF_acq) * weight                      # Noise in acquired LF
    mu_LF_acq, mu_LF_acq_max, mu_LF_acq_mask = compute_signal(LF_acq)                                 # signal in acquired LF
    snr_LF_acq                               = compute_snr(mu_LF_acq, sigma_LF_acq)                   # SNR in acquired LF
    print('\n SNR calculation in acquired Low Field')
    print('sigma_LF_acq: ', sigma_LF_acq)
    print('mu_LF_acq: ', mu_LF_acq)
    print('snr_LF_acq: ', snr_LF_acq)


    ## SNR calculation in B0 corrected acquired Low Field
    sigma_B0_corr                               = compute_noise_LF(B0_corr)                 # Noise in B0 corrected acquired LF
    mu_B0_corr, mu_B0_corr_max, mu_B0_corr_mask = compute_signal(B0_corr)                   # signal in B0 corrected acquired LF
    snr_B0_corr                                 = compute_snr(mu_B0_corr, sigma_B0_corr)    # SNR in B0 corrected acquired LF
    print('\n SNR calculation in B0_corrected: ')
    print('sigma_B0_corr: ', sigma_B0_corr)
    print('mu_B0_corr: ', mu_B0_corr)
    print('snr_B0_corr: ', snr_B0_corr)


    ## SNR calculation in High Field resized
    LF_sim                                   = Im_LFS_contrast_resize               # initializing LF_sim so that noise will be added into it to make final simulation
    sigma_LF_sim                             = compute_noise_HF(LF_sim)             # noise in LF_sim (this will be used in simulation)
    mu_LF_sim, mu_LF_sim_max, mu_LF_sim_mask = compute_signal(LF_sim)               # signal in LF sim
    snr_LF_sim                               = compute_snr(mu_LF_sim, sigma_LF_sim) # SNR in LF sim
    print('\n SNR calculation in High Field resized')
    print('sigma_LF_sim: ', sigma_LF_sim)
    print('mu_LF_sim: ', mu_LF_sim)
    print('snr_LF_sim: ', snr_LF_sim)
    # OrthoSlicer3D(LF_sim).show()


    ## SNR calculation in High Field Original
    sigma_HF                     = compute_noise_HF(Im_HF)              # noise in HF original
    mu_HF, mu_HF_max, mu_HF_mask = compute_signal(Im_HF)                # signal in HF original
    snr_HF                       = compute_snr(mu_HF, sigma_HF)         # SNR in HF original
    print('\n SNR calculation in High Field Original')
    print('sigma_HF: ', sigma_HF)
    print('mu_HF: ', mu_HF)
    print('snr_HF: ', snr_HF)


    ## initialize tolerance and noise factor
    # tolerance       = 0.05 * snr_LF_acq                                 # this is the tolerance for comparing the simulation to the target
    tolerance       = 0.05 * snr_B0_corr
    noise_fact_step = 0.5 * tolerance                                   # not used anymore
    noise_fact      = 0.2                                               # multiplication factor used while cutting out noise patch from noise matrix
    iter            = 0                                                 # counter


    ## Noise matrix
    LF_noise_matrix = compile_noise(noise_path, image_path_LF)
    # complile_noise creates noise matrix from which we are cutting out noise and then adding to simulation. It needs two inputs:
    # First input is the path to the folder which has all data for compiling noise i.e., noise_path.
    # Second input is the path to the file that you gave initially in the user input for image_path_LF.
    LF_acq_noise_patch = get_noise(LF_sim, LF_noise_matrix, noise_fact)
    # get_noise function cuts out a patch of size of LF_sim from noise matrix (LF_noise_matrix) and multiplies with noise_fact


    ## Initial mask of LF_sim
    mu_LF_sim_init, mu_LF_sim_max_init, LF_sim_mask = compute_signal(LF_sim, LF_sim_mask = 0)
    # Creates initial mask of LF_sim that is stored and then used in each iteration to maintain signal value
    print('\nmu_LF_sim_init: ', mu_LF_sim_init)


    print('\nSNR in HF resized image (snr_LF_sim): ', snr_LF_sim)
    print('Target SNR that we have to reach (snr_LF_acq): ', snr_LF_acq)
    print('Target SNR that we have to reach (snr_B0_corrected): ', snr_B0_corr)


    ## Noise addition in iteration
    while (snr_LF_sim - snr_B0_corr) > tolerance:
        iter = iter + 1

        # Get noise patch from noise matrix and add to LF_sim
        LF_acq_noise_patch = get_noise(LF_sim, LF_noise_matrix, noise_fact)
        LF_sim             = LF_sim + LF_acq_noise_patch

        # Use initial mask to get tissue
        LF_sim_tissue = LF_sim * LF_sim_mask

        # Calculating signal of tissue and then maintaining its value
        mu_LF_sim, mu_LF_sim_max, mu_LF_sim_mask = compute_signal(LF_sim_tissue, LF_sim_mask)
        LF_sim_tissue                            = np.divide(LF_sim_tissue, mu_LF_sim) * mu_LF_sim_init
        mu_LF_sim, mu_LF_sim_max, mu_LF_sim_mask = compute_signal(LF_sim_tissue, LF_sim_mask)

        # Getting background mask
        LF_sim_mask_bk = 1 - LF_sim_mask

        # Adding background mask with noise to simulation
        LF_sim = LF_sim_tissue + (LF_sim * LF_sim_mask_bk)

        # Computing SNR again after addition of noise
        sigma_LF_sim                             = compute_noise_HF(LF_sim)
        mu_LF_sim, mu_LF_sim_max, mu_LF_sim_mask = compute_signal(LF_sim, LF_sim_mask)
        snr_LF_sim                               = compute_snr(mu_LF_sim, sigma_LF_sim)

        # noise_fact = noise_fact - noise_fact_step * noise_fact        # updating the noise factor as well - not used anymore
        # noise_fact = noise_fact/iter                                  # not used anymore

        if np.mod(iter, 2) == 0:                                        # this will display the value of sigma_LF_sim after every second iteration
            print(iter)
            print('\t snr_LF_sim: ', snr_LF_sim)
            print('std: ', sigma_LF_sim)
            print('mu: ', mu_LF_sim)

    print('\nFinal matrix size of LF simulation (LF_sim): ', LF_sim.shape)
    print('Matrix size of acquired Low Field image (LF_acq) : ', LF_acq.shape)
    print('Matrix size of B0 corrected acquired Low Field image (B0_corr) : ', B0_corr.shape)

    print('\nFinal resolution after simulation: ', new_res_x, new_res_y, new_res_z)
    print('Target resolution that we had to reach (LF_acq) (same as B0 corrected): ', res_LF_acq)
    print('Original High Field Resolution:', res[0], res[1], res[2])

    print('\nNoise in HF after simulation (sigma_LF_sim): ', sigma_LF_sim)
    print('Target noise that we had to reach (sigma_LF_acq): ', sigma_LF_acq)
    print('Target noise that we had to reach (sigma_B0_corrected): ', sigma_B0_corr)

    print('\nSNR in HF after simulation (snr_LF_sim): ', snr_LF_sim)
    print('Target SNR that we had to reach (snr_LF_acq): ', snr_LF_acq)
    print('Target SNR that we had to reach (snr_B0_corrected): ', snr_B0_corr)



# Note:
# Theoretically, sigma_LT_theoretical should work because of the equation that we derived from the Andrew Webb's paper which says that noise in LF
# is 27.78 times the noise in HF. However, by using this sigma_LT_theoretical, the simulation is not working. Therefore, we used sigma_LT_practical.
# sigma_LT_practical is the noise from acquired LF image and then multiplied by weight factor of 5.
# This means to do the simulation, we have to reach 5 times more noise than the acquired LF image in order to get simulated LF image.
# The target should have been 27.78 * sigma_HF. But The actual target is becoming 5 * sigma_LF. This is a question to address.

# sigma_LFS_masked   = compute_noise(Im_LFS_contrast_resize)           # calculating noise in resized HF image (but this will not be used in simulation)
# sigma_LT_theoretical = sigma_LFS_masked * 27.78                 # From Andrew Webb paper - review eq. [] - codify this - not useful because 3T data background is masked

## This is an experiment function. Not required now
# sigma_n = get_sigma_n(mu_HF, mu_LF_acq, sigma_HF, sigma_LT_practical)


###########
# Viewing #
###########

if viewing == True:
    OrthoSlicer3D(Im_HF).show()
    OrthoSlicer3D(Im_LFS_contrast_resize).show()
    OrthoSlicer3D(LF_sim).show()
    OrthoSlicer3D(LF_acq).show()
    OrthoSlicer3D(B0_corr).show()

    # # Weighted averaging of B0 corrected image - for figure
    # w1 = 0.5
    # w2 = 0.5
    # i = 5
    # a = B0_corr[:, :, i] * w1
    # b = B0_corr[:, :, i + 1] * w2
    # B0_corr_new = (a + b)
    # B0_corr_new = np.abs(B0_corr_new)

    # # View two slices together
    # fig = plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(B0_corr_new, cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.imshow(LF_acq[:, :, 5], cmap="gray")
    # plt.show()

    # # Viewing single slice
    # fig = plt.figure()
    # plt.imshow(LF_acq[:, :, 5], cmap="gray")        # Im_HF - 6, LF_acq - 5, LF_sim - 6, B0_corr - 6
    # plt.show()

    # # Viewing single slice
    # fig = plt.figure()
    # plt.imshow(B0_corr[:, :, 14], cmap="gray")        # Im_HF - 14, LF_acq - 13, LF_sim - 14, B0_corr - 14
    # plt.show()

############
# Plotting #
############

# # For plotting figures
# for s in range(0,25):
# slice = Im_LFS_contrast_resize[:,:,3]
# fig = plt.figure()
# plt.imshow(slice, cmap="gray")
# plt.axis('off')
# plt.show()





