from colorama import Fore, Style
from sklearn.metrics import mean_squared_error
import itertools 
import numpy as np
from napari.utils import nbscreenshot
import napari
from nibabel.viewers import OrthoSlicer3D
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import scipy.ndimage as nd

def get_noise_LF(kSpaces, visualization=False,sigma_floor_frac=1e-6):
    # Check if the k-spaces are similar in shape and content to ensure they are repeated acquisitions
    if kSpaces.ndim != 4:
        raise ValueError("Expected kSpaces to be a 4D array")
    if kSpaces.shape[0] < 2:
        raise ValueError("At least two repeated acquisitions are required to estimate noise")

    # Check if the k-space data is consistent across acquisitions
    ref_shape = kSpaces.shape[1:]
    for i in range(1, kSpaces.shape[0]):
        if kSpaces[i].shape != ref_shape:
            raise ValueError(f"Inconsistent k-space shape at acquisition {i}")
        
    # Compute RMSE between all pairs of acquisitions to ensure similarity
    rmse_values = []
    for (i, j) in itertools.combinations(range(kSpaces.shape[0]), 2):
        rmse = np.sqrt(mean_squared_error(np.abs(kSpaces[i]).flatten(), np.abs(kSpaces[j]).flatten()))
        rmse_values.append(rmse)
    avg_rmse = np.mean(rmse_values)
    if avg_rmse > 1:  # Threshold can be adjusted based on expected noise level - 1/221
        raise ValueError("k-space acquisitions are not similar enough to estimate noise reliably")

    # preprocessing k-space data - remove spikes by clipping anything lower than 0 and higher than 99.5 percentile
    # Apply clipping to the magnitude of the kSpaces using masks
    kSpaces_mag = np.abs(kSpaces)
    clip_val = np.percentile(kSpaces_mag, 99.5)
    mask = kSpaces_mag > clip_val
    kSpaces_mag = np.clip(kSpaces_mag, a_min=0, a_max=clip_val)
    # Restore original phase
    kSpaces = kSpaces_mag * np.exp(1j * np.angle(kSpaces))

    # Compute successive differences - delta_kspace
    delta_kspace = np.diff(kSpaces, axis=0)
    # Estimate the variance of delta_kspace for each voxel
    variance = np.var(delta_kspace, axis=0, ddof=1)
    # The noise variance is half the variance of the differences
    noise_variance = variance / 2
    # The noise standard deviation is the square root of the noise variance
    sigma_map = np.sqrt(noise_variance)
    # post-processing - smooth the sigma map with a Gaussian filter to reduce high-frequency noise
    sigma_map = nd.gaussian_filter(sigma_map, sigma=1)
    # Generate a Gaussian noise map with the same shape as a single acquisition
        # floor
    global_med = np.median(sigma_map)
    sigma_map = np.maximum(sigma_map, global_med * sigma_floor_frac)
    

    if visualization is True:
        # Visualize the noise standard deviation map using napari's Orthoslicer
        # Visualize the noise standard deviation map using Orthoslicer
        OrthoSlicer3D(sigma_map).show()

        # Compute the inverse FFT to get the noise in image space
        noise_img_space = np.fft.ifftn(sigma_map)
        noise_img_space = np.fft.fftshift(noise_img_space)
        noise_img_space_abs = np.abs(noise_img_space)
        OrthoSlicer3D(noise_img_space_abs).show()

    return sigma_map

def get_noise_HF(LF_noise, LF_BW, HF_BW, LF_voxel, HF_voxel, B0_ratio=0.047/3, B0_alpha=1.0):
    # Scale the LF noise to HF noise based on bandwidth and voxel size ratios
    BW_ratio = np.sqrt(HF_BW / LF_BW)
    voxel_ratio = np.sqrt(np.prod(LF_voxel) / np.prod(HF_voxel))
    scaling_factor = BW_ratio * voxel_ratio * (B0_ratio ** (-B0_alpha))
    HF_noise = LF_noise * scaling_factor
    print(Fore.GREEN + f' Voxel ratio (sqrt(V_LF/V_HF)): {voxel_ratio}, Bandwidth ratio (sqrt(BW_HF/BW_LF)): {BW_ratio}, Overall scaling factor: {scaling_factor}' + Style.RESET_ALL)
    return HF_noise

def resize_noise_to_target(noise, target_shape):
    zoom_factors = [t / n for t, n in zip(target_shape, noise.shape)]
    resized_noise = zoom(noise, zoom_factors, order=3)  # Using bicubic interpolation
    print(Fore.GREEN + f'Resized noise from shape {noise.shape} to target shape {target_shape} using zoom factors {zoom_factors}' + Style.RESET_ALL)
    return resized_noise

def synthesize_complex_noise_map(shape, sigma_map, seed=None):
    rng = np.random.default_rng(seed)
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    return (real + 1j * imag) * sigma_map

# Simple SNR estimation utilities
def estimate_image_snr(image, signal_mask=None, noise_mask=None):
    """Estimate SNR = mean(signal) / std(noise).
    image: 3D or 2D magnitude image.
    If masks are None, heuristic masks are used: signal -> central cube; noise -> border region.
    Returns: snr (scalar) and (signal_mean, noise_std).
    """
    im = np.abs(image)
    if signal_mask is None:
        # central 20% cube as signal
        zc = im.shape[0] // 2 if im.ndim == 3 else None
        if im.ndim == 3:
            cz = int(im.shape[0]*0.2//2)
            cy = int(im.shape[1]*0.2//2)
            cx = int(im.shape[2]*0.2//2)
            signal_mask = np.zeros_like(im, dtype=bool)
            signal_mask[zc-cz:zc+cz, im.shape[1]//2-cy:im.shape[1]//2+cy, im.shape[2]//2-cx:im.shape[2]//2+cx] = True
        else:
            cy = int(im.shape[0]*0.2//2)
            cx = int(im.shape[1]*0.2//2)
            signal_mask = np.zeros_like(im, dtype=bool)
            signal_mask[im.shape[0]//2-cy:im.shape[0]//2+cy, im.shape[1]//2-cx:im.shape[1]//2+cx] = True
    if noise_mask is None:
        # border 10% as noise
        if im.ndim == 3:
            nz = int(im.shape[0]*0.1)
            ny = int(im.shape[1]*0.1)
            nx = int(im.shape[2]*0.1)
            noise_mask = np.zeros_like(im, dtype=bool)
            noise_mask[:nz, :, :] = True
            noise_mask[-nz:, :, :] = True
            noise_mask[:, :ny, :] = True
            noise_mask[:, -ny:, :] = True
            noise_mask[:, :, :nx] = True
            noise_mask[:, :, -nx:] = True
            # avoid overlap
            noise_mask = noise_mask & (~signal_mask)
        else:
            ny = int(im.shape[0]*0.1)
            nx = int(im.shape[1]*0.1)
            noise_mask = np.zeros_like(im, dtype=bool)
            noise_mask[:ny, :] = True
            noise_mask[-ny:, :] = True
            noise_mask[:, :nx] = True
            noise_mask[:, -nx:] = True
            noise_mask = noise_mask & (~signal_mask)

    sig = im[signal_mask].mean()
    noise_std = im[noise_mask].std(ddof=1)
    snr = sig / (noise_std + 1e-12)
    return snr, sig, noise_std, signal_mask, noise_mask

def get_im(kspace):
    return np.fft.ifft2(np.fft.ifftshift(kspace))

def get_kspace(image):
    return np.fft.fftshift(np.fft.fft2(image))