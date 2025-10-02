from colorama import Fore, Style

# ... rest of your imports ...

# Replace all print statements like this:
# print('Matrix size of High Field image: ', Im_HF.shape)
# =>
# print(Fore.GREEN + 'Matrix size of High Field image: ', Im_HF.shape, Style.RESET_ALL)

# Here is your code with updated print statements:

#######################################################
## Low Field Simulation of High Field Phantom images ##
#######################################################

import numpy as np
import pydicom as pyd
import matplotlib
import matplotlib.pyplot as plt
import keaDataProcessing as keaProc
# from pyfeats import glcm_features
# from roipoly import RoiPoly
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
from colorama import Fore, Style

##############
# User input #
##############

nifti_input     = False         # make it true if you want to use NIFTI data
dicom_input     = True          # make it true if you want to use DICOM data - for NHP
IMA_dicom_input = False         # make it true if you want to use DICOM IMA data - for phantom

contrast_component   = False    # make it true if you want to add contrast component
resolution_component = True     # make it true if you want to add resolution component
snr_component        = True     # make it true if you want to add SNR component

viewing = True                  # make it true to see output
# Sequence of output - (1)HF, (2)HF resized, (3)LF simulated, (4)LF acquired
saving = False

# New Resolution - change these to get desired resolution
new_res_x = 2
new_res_y = 2
new_res_z = 5

# Threshold for segmenting phantom - don't change until required. It affects SNR component.
thresh = 0.2

## All paths
image_path_HF = './data_sim_check/3T'
LF_sim_saving_path = './data_sim_check/LF_simulated_2x2x5'
image_path_LF = './data_sim_check/47mT/data.3d'
image_path_LF_dir = './data_sim_check/47mT'
acqu_path     = './data_sim_check/47mT/acqu.par'
# noise_path    = './data_sim_check/47mT/noise.par'
image_path_LF_noise = './6_data_for_noise/data'

############################
# Initialization of values #
############################

gamma_bar = 42.57 * (10**6)

a_WM = 0.71
a_GM = 1.16
b_WM = 0.382
b_GM = 0.376

B0_HF = 3
B0_LF = 0.05

TR = 500

T1_GM_LF_mean = 335
T1_GM_LF_std  = 25

T1_WM_LF_mean = 280
T1_WM_LF_std  = 15

T1_GM_HF_mean = 1300
T1_GM_HF_std  = 50

T1_WM_HF_mean = 832
T1_WM_HF_std  = 20

##################
# Importing data #
##################

## Import High field image - DICOM
if dicom_input == True:
    Im_HF, ds, Nz_HF = read_dicoms(image_path_HF)
    Im_HF = Im_HF.astype(float)
    print(Fore.GREEN + 'Matrix size of High Field image: ', Im_HF.shape, Style.RESET_ALL)

## Import High Field image - DICOM IMA
if IMA_dicom_input == True:
    Im_HF, ds, Nz_HF = read_dicoms_IMA(image_path_HF)
    Im_HF = Im_HF.astype(float)
    print(Fore.GREEN + 'Matrix size of High Field image: ', Im_HF.shape, Style.RESET_ALL)

## Import High Field image - NIFTI
if nifti_input == True:
    Im_HF, hdr = read_nifti(image_path_HF)
    print(Fore.GREEN + 'Matrix size of High Field image: ', Im_HF.shape, Style.RESET_ALL)

ImageScanParams = keaProc.readPar(acqu_path)
kSpace          = keaProc.readKSpace(image_path_LF)
LF_acq          = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(kSpace)))
LF_acq          = np.abs(LF_acq)
fov_LF_acq      = ImageScanParams.get('FOV')
matrix_LF_acq   = LF_acq.shape
res_LF_acq      = np.divide(fov_LF_acq,matrix_LF_acq)
print(Fore.CYAN + 'Matrix size of acquired Low Field image: ', matrix_LF_acq, Style.RESET_ALL)
print(Fore.CYAN + 'FOV of acquired LF: ', fov_LF_acq, Style.RESET_ALL)
print(Fore.CYAN + 'Resolution of acquired LF: ', res_LF_acq, Style.RESET_ALL)

Im_HF = normalize(Im_HF)

# Display a collage of 16 continuous in-plane slices for benchmarking

def show_inplane_collage(volume, title_prefix="", axis=2):
    # axis=2 means axial (z), axis=0 sagittal, axis=1 coronal
    mid = volume.shape[axis] // 2
    # Get 16 slices centered at the middle
    half = 8
    start = max(0, mid - half + 1)
    end = min(volume.shape[axis], start + 16)
    slices = [volume.take(i, axis=axis) for i in range(start, end)]

    fig, axs = plt.subplots(4, 4, figsize=(16, 16))
    axs = axs.flatten()
    for idx, (slc, ax) in enumerate(zip(slices, axs)):
        ax.imshow(slc.T, cmap="gray", origin="lower")
        ax.set_title(f"{title_prefix} slice {start + idx}")
        ax.axis('off'
        )
    for ax in axs[len(slices):]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Show 3T (High Field) in-plane slices (axial)
show_inplane_collage(Im_HF, title_prefix="3T", axis=2)

# Show 47mT (Low Field acquired) in-plane slices (axial)
show_inplane_collage(LF_acq, title_prefix="47mT", axis=2)



############
# Contrast #
############

if contrast_component == True:
    mask, mask_header, Nz_mask = read_dicoms(mask_path)
    mask_CSF = np.zeros(mask.shape)
    mask_CSF[mask==2] = 1
    Im_LF_CSF = np.multiply(Im_HF, mask_CSF)
    mask_GM = np.zeros(mask.shape)
    mask_GM[mask==3] = 1
    Im_HF_GM = np.multiply(Im_HF, mask_GM)
    mask_WM = np.zeros(mask.shape)
    mask_WM[mask==4] = 1
    Im_HF_WM = np.multiply(Im_HF, mask_WM)
    Im_LF_GM = Contrast_map(TR,T1_GM_LF_mean,T1_GM_HF_mean,Im_HF_GM)
    Im_LF_WM = Contrast_map(TR,T1_WM_LF_mean,T1_WM_HF_mean,Im_HF_WM)
    Im_LF_contrast = Im_LF_WM + Im_LF_GM + Im_LF_CSF
    print(Fore.YELLOW + 'Shape of LF contrasts combined: ', Im_LF_contrast.shape, Style.RESET_ALL)

##########################
# Down Res of High field #
##########################

if resolution_component == True:
    if 'Im_LF_contrast' in globals():
        input_for_down_res = Im_LF_contrast
        print(Fore.YELLOW + '\n Contrast component detected and used' + Style.RESET_ALL)
        print(Fore.YELLOW + '\n Down Resolution started' + Style.RESET_ALL)
    else:
        input_for_down_res = Im_HF
        print(Fore.YELLOW + '\n Contrast component not detected' + Style.RESET_ALL)
        print(Fore.YELLOW + '\n Down Resolution started' + Style.RESET_ALL)

    if nifti_input == True:
        dim     = hdr['dim']
        pixdim  = hdr['pixdim']
        matrix  = [dim[1], dim[2], dim[3]]
        res     = [pixdim[1], pixdim[2], pixdim[3]]
        Im_LFS_contrast_resize  = resize_data(Im_HF,res,matrix)
        print(Fore.YELLOW + 'Original High Field Matrix size:', matrix[0], matrix[1], matrix[2], Style.RESET_ALL)
        print(Fore.YELLOW + 'Original High Field Resolution:', res[0], res[1], res[2], Style.RESET_ALL)
        print(Fore.YELLOW + 'Matrix Size after Down Res: ', Im_LFS_contrast_resize.shape, Style.RESET_ALL)

    if dicom_input or IMA_dicom_input == True:
        matrix                 = [np.max(ds.Rows), np.max(ds.Columns), Nz_HF]
        res                    = [ds.PixelSpacing[0], ds.PixelSpacing[1], ds.SliceThickness]
        Im_LFS_contrast_resize = resize_data(input_for_down_res, res, matrix, new_res_x, new_res_y, new_res_z)
        print(Fore.YELLOW + 'Original High Field Matrix size:', matrix[0], matrix[1], matrix[2], Style.RESET_ALL)
        print(Fore.YELLOW + 'Original High Field Resolution:', res[0], res[1], res[2], Style.RESET_ALL)
        print(Fore.YELLOW + 'Matrix Size after Down Res: ', Im_LFS_contrast_resize.shape, Style.RESET_ALL)

    print(Fore.YELLOW + 'Down Resolution finished' + Style.RESET_ALL)

###################
####### SNR #######
###################

if snr_component == True:
    if 'Im_LFS_contrast_resize' in globals():
        input_for_snr = Im_LFS_contrast_resize
        print(Fore.MAGENTA + '\n Down res component detected and used' + Style.RESET_ALL)
        print(Fore.MAGENTA + '\n SNR component started' + Style.RESET_ALL)
    else:
        input_for_snr = Im_HF
        print(Fore.MAGENTA + '\n Down res component not detected' + Style.RESET_ALL)
        print(Fore.MAGENTA + '\n SNR component started' + Style.RESET_ALL)

    Im_LFS_contrast_resize = normalize(input_for_snr)
    LF_acq                 = normalize(LF_acq)

    weight = 1
    sigma_LF_acq                             = compute_noise_LF(LF_acq) * weight
    mu_LF_acq, mu_LF_acq_max, mu_LF_acq_mask = compute_signal(LF_acq)
    snr_LF_acq                               = compute_snr(mu_LF_acq, sigma_LF_acq)
    print(Fore.BLUE + '\n SNR calculation in acquired Low Field' + Style.RESET_ALL)
    print(Fore.BLUE + 'sigma_LF_acq: ', sigma_LF_acq, Style.RESET_ALL)
    print(Fore.BLUE + 'mu_LF_acq: ', mu_LF_acq, Style.RESET_ALL)
    print(Fore.BLUE + 'snr_LF_acq: ', snr_LF_acq, Style.RESET_ALL)

    LF_sim                                   = Im_LFS_contrast_resize
    sigma_LF_sim                             = compute_noise_HF(LF_sim)
    sigma_LF_sim                             = 0.0003586680668437035
    mu_LF_sim, mu_LF_sim_max, mu_LF_sim_mask = compute_signal(LF_sim)
    snr_LF_sim                               = compute_snr(mu_LF_sim, sigma_LF_sim)
    print(Fore.BLUE + '\n SNR calculation in High Field resized' + Style.RESET_ALL)
    print(Fore.BLUE + 'sigma_LF_sim: ', sigma_LF_sim, Style.RESET_ALL)
    print(Fore.BLUE + 'mu_LF_sim: ', mu_LF_sim, Style.RESET_ALL)
    print(Fore.BLUE + 'snr_LF_sim: ', snr_LF_sim, Style.RESET_ALL)

    sigma_HF                     = compute_noise_HF(Im_HF)
    mu_HF, mu_HF_max, mu_HF_mask = compute_signal(Im_HF)
    snr_HF                       = compute_snr(mu_HF, sigma_HF)
    print(Fore.BLUE + '\n SNR calculation in High Field Original' + Style.RESET_ALL)
    print(Fore.BLUE + 'sigma_HF: ', sigma_HF, Style.RESET_ALL)
    print(Fore.BLUE + 'mu_HF: ', mu_HF, Style.RESET_ALL)
    print(Fore.BLUE + 'snr_HF: ', snr_HF, Style.RESET_ALL)

    tolerance       = 0.05 * snr_LF_acq
    noise_fact_step = 0.5 * tolerance
    noise_fact      = 0.2
    iter            = 0

    LF_noise_matrix = compile_noise(image_path_LF_dir, image_path_LF)
    LF_acq_noise_patch = get_noise(LF_sim, LF_noise_matrix, noise_fact)

    mu_LF_sim_init, mu_LF_sim_max_init, LF_sim_mask = compute_signal(LF_sim, LF_sim_mask = 0)
    print(Fore.MAGENTA + '\nmu_LF_sim_init: ', mu_LF_sim_init, Style.RESET_ALL)

    print(Fore.MAGENTA + '\nSNR in HF resized image (snr_LF_sim): ', snr_LF_sim, Style.RESET_ALL)
    print(Fore.MAGENTA + 'Target SNR that we have to reach (snr_LF_acq): ', snr_LF_acq, Style.RESET_ALL)

    while (snr_LF_sim - snr_LF_acq) > tolerance:
        iter = iter + 1
        LF_acq_noise_patch = get_noise(LF_sim, LF_noise_matrix, noise_fact)
        LF_sim             = LF_sim + LF_acq_noise_patch
        LF_sim_tissue = LF_sim * LF_sim_mask
        mu_LF_sim, mu_LF_sim_max, mu_LF_sim_mask = compute_signal(LF_sim_tissue, LF_sim_mask)
        LF_sim_tissue                            = np.divide(LF_sim_tissue, mu_LF_sim) * mu_LF_sim_init
        mu_LF_sim, mu_LF_sim_max, mu_LF_sim_mask = compute_signal(LF_sim_tissue, LF_sim_mask)
        LF_sim_mask_bk = 1 - LF_sim_mask
        LF_sim = LF_sim_tissue + (LF_sim * LF_sim_mask_bk)
        sigma_LF_sim                             = compute_noise_HF(LF_sim)
        mu_LF_sim, mu_LF_sim_max, mu_LF_sim_mask = compute_signal(LF_sim, LF_sim_mask)
        snr_LF_sim                               = compute_snr(mu_LF_sim, sigma_LF_sim)

        if np.mod(iter, 2) == 0:
            print(Fore.RED + str(iter) + Style.RESET_ALL)
            print(Fore.RED + '\t snr_LF_sim: ', snr_LF_sim, Style.RESET_ALL)
            print(Fore.RED + 'std: ', sigma_LF_sim, Style.RESET_ALL)
            print(Fore.RED + 'mu: ', mu_LF_sim, Style.RESET_ALL)

    print(Fore.GREEN + '\nFinal matrix size of LF simulation (LF_sim): ', LF_sim.shape, Style.RESET_ALL)
    print(Fore.GREEN + 'Matrix size of acquired Low Field image (LF_acq) : ', LF_acq.shape, Style.RESET_ALL)
    print(Fore.GREEN + '\nFinal resolution after simulation: ', new_res_x, new_res_y, new_res_z, Style.RESET_ALL)
    print(Fore.GREEN + 'Target resolution that we had to reach (LF_acq): ', res_LF_acq, Style.RESET_ALL)
    print(Fore.GREEN + 'Original High Field Resolution:', res[0], res[1], res[2], Style.RESET_ALL)
    print(Fore.GREEN + '\nNoise in HF after simulation (sigma_LF_sim): ', sigma_LF_sim, Style.RESET_ALL)
    print(Fore.GREEN + 'Target noise that we had to reach (sigma_LF_acq): ', sigma_LF_acq, Style.RESET_ALL)
    print(Fore.GREEN + '\nSNR in HF after simulation (snr_LF_sim): ', snr_LF_sim, Style.RESET_ALL)
    print(Fore.GREEN + 'Target SNR that we had to reach (snr_LF_acq): ', snr_LF_acq, Style.RESET_ALL)

###########
# Viewing #
###########

if viewing == True:
    OrthoSlicer3D(LF_sim).show()
    test = LF_sim
    fig = plt.figure()
    plt.imshow(LF_sim[:, :, 4], cmap="gray")
    plt.show()

##########
# Saving #
##########

if saving == True:
    LF_sim_slice = LF_sim[:, :, slice_number]
    np.save(LF_sim_saving_path, LF_sim_slice)
