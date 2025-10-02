# This file outlines the steps to produce a simulated low-field MRI image from a high-field MRI image.
# It has the following steps:
# 1. Load high field MRI image
# 2. If available, load a low field MRI image for reference
# 3. Derive or assume the SD of the HF MRI image noise
# 4. Extract noise patches from the LF reference image (if available) or from the preloaded LF data for noise
# 5. Compute the target SNR based on the HF image noise and the desired LF image noise
# 6. Iteratively add noise patches to the HF image until the target SNR is reached
# 7. Visualize the results and show the SNR progression
# 8. Save the simulated LF image

from scipy.ndimage import zoom
from colorama import Fore, Style
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
import random   
from skimage.util import view_as_windows
from skimage import exposure
from scipy.stats import norm
import warnings


warnings.filterwarnings("ignore", category=UserWarning)
import argparse
import sys
import logging
from datetime import datetime
import time
import pydicom
from glob import glob
import keaDataProcessing as keaProc
from utils import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LF_sim_worker:
    def __init__(self, hf_path=None, lf_path=None, lf_reference_path=None, target_resolution=[1, 1, 2], output_folder=None):
        self.hf_path = hf_path
        self.lf_path = lf_path
        self.lf_reference_path = lf_reference_path
        self.hf_image = None
        self.lf_image = None
        self.target_resolution = target_resolution  # in mm, can be adjusted based on the LF system
        self.output_folder = output_folder

    def load_hf_dicom_series(self, dicom_folder):

        dicom_files = sorted(glob(os.path.join(dicom_folder, '*.dcm')))
        if not dicom_files:
            logger.error(f"No DICOM files found in {dicom_folder}")
            return None

        slices = [pydicom.dcmread(f) for f in dicom_files]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        img_shape = list(slices[0].pixel_array.shape)
        img_shape.append(len(slices))
        hf_img = np.zeros(img_shape, dtype=slices[0].pixel_array.dtype)
        for i, s in enumerate(slices):
            hf_img[:, :, i] = s.pixel_array
        self.hf_image = hf_img
        self.hf_og = hf_img.copy()  # keep a copy of the original HF image
        logger.info(f"Loaded HF DICOM series from {dicom_folder} with shape {hf_img.shape}")
        # Also store the pixel spacing and slice thickness
        self.HF_voxel = [float(slices[0].PixelSpacing[0]), float(slices[0].PixelSpacing[1]), float(slices[0].SliceThickness)]
        print(Fore.CYAN + 'Voxel size of HF image: ', self.HF_voxel, Style.RESET_ALL)
        # Estimate the bandwidth from DICOM tags if available
        if 'PixelBandwidth' in slices[0]:
            self.HF_BW = float(slices[0].PixelBandwidth)  # in Hz/pixel
        else:
            self.HF_BW = 600  # Hz/pixel Assume a default value if not available
        return hf_img

    def load_lf_3d_file(self, lf_path):
        if lf_path != 'stored':
            # Assuming .3d is a simple binary float32 file with known dimensions
            # You may need to adapt this to your .3d file format
            acqu_path = lf_path + '/acqu.par'
            image_path_LF = lf_path + '/data.3d'
            ImageScanParams = keaProc.readPar(acqu_path)
            self.LF_ref_kSpace = keaProc.readKSpace(image_path_LF)
            LF_acq = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(self.LF_ref_kSpace)))
            self.LF_ref_im = np.abs(LF_acq)
            self.fov_LF_ref_acq = ImageScanParams.get('FOV')
            self.matrix_LF_ref_acq = LF_acq.shape
            self.res_LF_ref_acq = np.divide(self.fov_LF_ref_acq, self.matrix_LF_ref_acq)
            print(Fore.CYAN + 'Matrix size of acquired Low Field image: ', self.matrix_LF_ref_acq, Style.RESET_ALL)
            print(Fore.CYAN + 'FOV of acquired LF: ', self.fov_LF_ref_acq, Style.RESET_ALL)
            print(Fore.CYAN + 'Resolution of acquired LF: ', self.res_LF_ref_acq, Style.RESET_ALL)

        else: # need to read the LF data from a directory that contains multiple .3d files and concatenate their image spaces to obtain noise patches
            lf_path = './6_data_for_noise/noise_niv'
            # Obtain the voxel and bandwidth information from one of the acqu.par files
            lf_path_acqu = './6_data_for_noise/acqu_niv/acqu1.par'  # This needs to be updated after the noise acquisition experiment
            ImageScanParams = keaProc.readPar(lf_path_acqu)
            self.LF_voxel = [float(ImageScanParams.get('FOVread')) / int(ImageScanParams.get('nrPnts')),
                             float(ImageScanParams.get('FOVphase1')) / int(ImageScanParams.get('nPhase1')),
                             float(ImageScanParams.get('FOVphase2')) / int(ImageScanParams.get('nPhase2'))]
            self.LF_BW = float(ImageScanParams.get('bandwidth')) * 1000 / int(ImageScanParams.get('nrPnts'))  # in Hz/pixel
            print(Fore.CYAN + 'Voxel size of LF image: ', self.LF_voxel, Style.RESET_ALL)
            print(Fore.CYAN + 'Bandwidth of LF image: ', self.LF_BW, Style.RESET_ALL)

            lf_files = sorted(glob(os.path.join(lf_path, '*.3d')))
            # Estimating noise using the repeated acquisitions available from data acquired for noise
            # Read all the .3d files and concatenate the kspace data into one 4D array
            kSpaces = []
            for lf_file in lf_files:
                kSpace = keaProc.readKSpace(lf_file)
                kSpaces.append(kSpace)
            self.kSpaces = np.array(kSpaces)


    def load_data(self):
        if self.hf_path:
            self.load_hf_dicom_series(self.hf_path)
        if self.lf_path:
            self.load_lf_3d_file(self.lf_path)
        if self.lf_reference_path:
            self.load_lf_3d_file(self.lf_reference_path)
            
    def resize_hf_to_target_resolution(self):
        # Compute the target shape based on the target resolution and current voxel size
        target_shape = [
            int(round(self.hf_image.shape[i] * self.HF_voxel[i] / self.target_resolution[i]))
            for i in range(3)
        ]
        zoom_factors = [t / s for t, s in zip(target_shape, self.hf_image.shape)]
        self.hf_image = zoom(self.hf_image, zoom_factors, order=3)  # Using bicubic interpolation
        self.y_train = self.hf_image.copy()  # Save the resized HF image as ground truth for training

        # Now resize the self.hf to match the LF reference image before adding noise
        if self.lf_reference_path:
            target_shape = self.LF_ref_im.shape
            zoom_factors = [t / s for t, s in zip(target_shape, self.hf_image.shape)]
            self.hf_image = zoom(self.hf_image, zoom_factors, order=3)  # Using bicubic interpolation
        print(Fore.GREEN + f'Resized HF image to target shape {target_shape} using zoom factors {zoom_factors}' + Style.RESET_ALL)
        return self.hf_image

    def compute_noise(self, visualization=False):
            # Obtain the noise component using the repeated acquisitions available from data acquired for noise
            self.LF_sigma_map = get_noise_LF(self.kSpaces, visualization=visualization)
            print(Fore.GREEN + 'Noise component extracted from LF data' + Style.RESET_ALL)
            self.HF_sigma_map = get_noise_HF(self.LF_sigma_map, self.LF_BW, self.HF_BW, self.LF_voxel, self.HF_voxel)
            self.HF_sigma_map_resized = resize_noise_to_target(self.HF_sigma_map, self.hf_image.shape)
            print(Fore.GREEN + 'Resized HF noise to match HF image shape' + Style.RESET_ALL)
            # Synthesize complex noise map
            self.HF_noise_complex = synthesize_complex_noise_map(self.hf_image.shape, self.HF_sigma_map_resized, seed=0)

    def get_stats(self, image=None):
        if image is None:
            image = self.hf_image
        snr, sig, noise_std, _, _ = estimate_image_snr(image)
        print(Fore.CYAN + f'Image stats - SNR: {snr:.2f}, Signal mean: {sig:.2f}, Noise std: {noise_std:.2f}' + Style.RESET_ALL)
        return snr, sig, noise_std
    
    def add_noise_to_hf_image(self, alpha=0.5, max_iterations=20, snr_tolerance=0.1, visualization =False):
        if self.hf_image is None or self.HF_noise_complex is None:
            logger.error("HF image or noise not loaded.")
            return
        hf_k = get_kspace(self.hf_image)
        hf_k_noisy = hf_k.copy()


        # Initial SNR of the HF image
        initial_snr, _, _, _, _ = estimate_image_snr(self.hf_image)
        print(Fore.CYAN + f'Initial HF Image SNR: {initial_snr:.2f}' + Style.RESET_ALL)

        lf_sim_target_snr, _, _, _, _ = estimate_image_snr(self.LF_ref_im)
        print(Fore.CYAN + f'Target LF Image SNR (from reference LF image): {lf_sim_target_snr:.2f}' + Style.RESET_ALL)

    
        current_snr = initial_snr
        iteration = 0
        snr_progression = [current_snr]

        while abs(current_snr - lf_sim_target_snr) > snr_tolerance and iteration < max_iterations:
            # Add a fraction of the noise to the k-space
            # noise = alpha * self.HF_noise_complex  # Should we add the same noise repeatedly or different noise each time?
            noise = alpha * synthesize_complex_noise_map(self.hf_image.shape, self.HF_sigma_map_resized, seed=iteration)
            hf_k_noisy = hf_k_noisy + noise
            im_noisy = np.abs(get_im(hf_k_noisy))
            current_snr, _, _, _, _ = estimate_image_snr(im_noisy)
            self.im_noisy = im_noisy
            snr_progression.append(current_snr)
            iteration += 1
            print(Fore.YELLOW + f'Iteration {iteration}: Current SNR: {current_snr:.2f}' + Style.RESET_ALL)

            if visualization and (iteration % 50 == 0 or iteration == max_iterations or abs(current_snr - lf_sim_target_snr) <= snr_tolerance):
                mid_slice_hf = self.y_train.shape[2] // 2
                mid_slice_lf = self.LF_ref_im.shape[2] // 2
                mid_slice_lf_noisy = im_noisy.shape[2] // 2
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(self.y_train[:, :, mid_slice_hf], cmap='gray')
                axs[0].set_title('Clean HF (mid slice)')
                axs[0].axis('off')
                axs[1].imshow(im_noisy[:, :, mid_slice_lf_noisy], cmap='gray')
                axs[1].set_title(f'Noisy HF (iter {iteration})')
                axs[1].axis('off')
                axs[2].imshow(self.LF_ref_im[:, :, mid_slice_lf], cmap='gray')
                axs[2].set_title('Reference LF (mid slice)')
                axs[2].axis('off')
                plt.suptitle(f'Iteration {iteration}: SNR={current_snr:.2f} (Target: {lf_sim_target_snr:.2f})')
                plt.tight_layout()
                plt.show()
    
    def visualize_central_slices(self):
        fig, axs = plt.subplots(1, 5, figsize=(20, 5))
        titles = ['Original HF (hf_og)', 'Resized HF (y_train)', 'Noisy LF Sim (im_noisy)', 'Upsampled LF Sim (x_train)', 'Reference LF (LF_ref_im)']
        images = [
            self.hf_og[:, :, self.hf_og.shape[2] // 2],
            self.y_train[:, :, self.y_train.shape[2] // 2],
            self.im_noisy[:, :, self.im_noisy.shape[2] // 2],
            self.x_train[:, :, self.x_train.shape[2] // 2],
            self.LF_ref_im[:, :, self.LF_ref_im.shape[2] // 2]
        ]
        for ax, img, title in zip(axs, images, titles):
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    def resize_noisy_sim_to_hf_resolution(self):
        if self.hf_image is None:
            logger.error("HF image not loaded.")
            return
        # Resample back to original HF resolution
        target_shape = self.y_train.shape
        zoom_factors = [t / s for t, s in zip(target_shape, self.im_noisy.shape)]
        self.simulated_lf_image_upsampled = zoom(self.im_noisy, zoom_factors, order=3)  # Using bicubic interpolation
        print(Fore.GREEN + f'Resized simulated LF image back to target HF shape {target_shape} using zoom factors {zoom_factors}' + Style.RESET_ALL)
        
        # Save the simulated LF image as a NIfTI file
        self.x_train = self.simulated_lf_image_upsampled.copy()  # Save the simulated LF image for training
        return self.simulated_lf_image_upsampled

    def save_data_for_training(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        # Save x_train and y_train as .npy files
        # Combine x_train and y_train into a dictionary and save as a single .npy file
        data_dict = {'x_train': self.x_train, 'y_train': self.y_train}
        np.save(os.path.join(self.output_folder, 'train_data.npy'), data_dict)
        return self.hf_og, self.x_train, self.y_train
    
# Perform the simulation below with the steps outlined above
hf_path = './data_sim_check/3T'  # Replace with your HF DICOM series folder
lf_noise_path = 'stored'  # Replace with your LF .3d file or folder - with repeated noise acquisitions
lf_reference_path = './data_sim_check/47mT'  # Replace with your LF reference image path if available - to get target SNR
output_folder = hf_path + '_simulated_LF'

Halbach_sim_worker = LF_sim_worker(hf_path, lf_noise_path, lf_reference_path, target_resolution=[1, 1, 2], output_folder=output_folder)
Halbach_sim_worker.load_data()
Halbach_sim_worker.resize_hf_to_target_resolution()
Halbach_sim_worker.compute_noise(visualization=False)
Halbach_sim_worker.add_noise_to_hf_image(alpha = 0.5, max_iterations=800, snr_tolerance=0.1, visualization=False)
Halbach_sim_worker.resize_noisy_sim_to_hf_resolution() # resample back to target resolution specified - usually 1 x 1 x 2 mm_cubed for NiV
[hf_og, x_train, y_train] = Halbach_sim_worker.save_data_for_training()  # saves the x_train and y_train as .npy files in the output folder
print(Fore.GREEN + f'Saved training data in {Halbach_sim_worker.output_folder}' + Style.RESET_ALL)
print(Fore.CYAN + f'Original HF image shape: {hf_og.shape}, Simulated LF image shape: {x_train.shape}, Resized HF image shape: {y_train.shape}, Target resolution: {Halbach_sim_worker.target_resolution}' + Style.RESET_ALL)
Halbach_sim_worker.visualize_central_slices()