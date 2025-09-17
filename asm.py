#!/usr/bin/env python3
import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import os,sys
import subprocess
import glob

def angular_spectrum_propagation(hologram, wavelength, distance, dx, dy):
    """
    Performs Angular Spectrum Method (ASM) propagation in Python.

    Args:
        hologram (np.ndarray): The initial complex-valued hologram field (2D array).
        wavelength (float): Wavelength of the light (in meters).
        distance (float): Propagation distance (in meters).
        dx (float): Pixel pitch in the x-direction (in meters).
        dy (float): Pixel pitch in the y-direction (in meters).

    Returns:
        np.ndarray: The propagated complex-valued field.
    """
    ny, nx = hologram.shape
    
    # Calculate spatial frequency coordinates
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    
    # Create the meshgrid for spatial frequencies
    FX, FY = np.meshgrid(fx, fy)
    
    # Calculate the spatial frequency squared
    FX2_plus_FY2 = FX**2 + FY**2
    
    # Calculate the propagation kernel H in the frequency domain
    k = 2 * np.pi / wavelength
    
    # The evanescent waves (negative square root) are handled by setting the kernel to 0
    # for frequencies beyond the cutoff, but numpy handles the complex numbers for us.
    propagator_exponent = k * np.sqrt(1.0 - (wavelength * FX)**2 - (wavelength * FY)**2)
    H = np.exp(1j * propagator_exponent * distance)
    
    # Transform the initial field to the frequency domain
    U0_freq = fftshift(fft2(ifftshift(hologram)))
    
    # Multiply by the propagation kernel
    U_freq = U0_freq * H
    
    # Inverse transform to get the propagated field in the spatial domain
    U_prop = fftshift(ifft2(ifftshift(U_freq)))
    
    return U_prop

def calculate_dihm_propagation_kernel(hologram, wavelength, distance_1, distance_2, dx, dy):
    """
    Calculates the propagation kernel for a double-heterodyne (DIHM) setup
    by performing two successive ASM propagations.

    Args:
        hologram (np.ndarray): The initial complex-valued hologram field (2D array).
        wavelength (float): Wavelength of the light (in meters).
        distance_1 (float): First propagation distance (in meters).
        distance_2 (float): Second propagation distance (in meters).
        dx (float): Pixel pitch in the x-direction (in meters).
        dy (float): Pixel pitch in the y-direction (in meters).
        
    Returns:
        np.ndarray: The numerically reconstructed DIHM hologram.
    """
    # First propagation
    propagated_field_1 = angular_spectrum_propagation(hologram, wavelength, distance_1, dx, dy)
    
    # Second propagation
    # In a DIHM setup, the second distance might be negative for back-propagation
    # or involve a different setup. This example performs another forward propagation
    # for illustration. For reconstruction, you typically propagate forward and then
    # backpropagate to the object plane. Backpropagation is achieved by using a 
    # negative propagation distance.
    propagated_field_2 = angular_spectrum_propagation(propagated_field_1, wavelength, distance_2, dx, dy)
    
    return propagated_field_1

# Example Usage:
if __name__ == '__main__':
    # Define parameters for a hypothetical hologram
    destination_path_safe = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.join('images', 'hologram4_resized.png'))
    hologram = cv2.imread(destination_path_safe, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    hologram = hologram[:512, :512]  # Ensure it's 512x512 for this example
    nx, ny = 512, 512
    dx, dy = 2e-6, 2e-6  # 5 micrometers pixel pitch
    wavelength = 633e-9 # 633 nm (red laser)
    distance_1 = 1e-3  # 1 mm
    distance_2 = -1e-3 # -1 mm (back-propagation)
    
    # Create a dummy hologram (e.g., a simple phase object)
    x = np.arange(0, nx) * dx
    y = np.arange(0, ny) * dy
    X, Y = np.meshgrid(x, y)

    """
    # Create a simple test object (e.g., a circular phase shift)
    center_x, center_y = nx * dx / 2, ny * dy / 2
    radius = 50 * dx
    mask = ((X - center_x)**2 + (Y - center_y)**2) < radius**2
    phase_shift = np.zeros((ny, nx))
    phase_shift[mask] = np.pi / 2
    
    hologram = np.exp(1j * phase_shift)
    """
    # os.system("mkdir -p recon")
    recon_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'recon')
    # os.makedirs(recon_path, exist_ok=True)
    # recon_delete_path = os.path.join(recon_path, f'recon*.png')
    # os.system("rm -f recon/recon*.png")
    # subprocess.run('del recon_delete_path', shell=True, check=True)
    frameno = 0
    for distance in np.arange(0.08e-3, 0.30e-3, 0.003e-3):

        # Calculate the double-heterodyne propagation
        reconstructed_hologram = calculate_dihm_propagation_kernel(
            hologram, wavelength, distance, -distance, dx, dy
        )
        frameno += 1
        print(f"Distance: {distance*1e3:.3f} mm")
        recon_write_path = os.path.join(recon_path, f'recon{frameno:05d}.png')
        cv2.imwrite(recon_write_path, np.abs(reconstructed_hologram) / np.max(np.abs(reconstructed_hologram)) * 255)
        
    # Print the reconstructed field's shape and some values
    print("Original hologram shape:", hologram.shape)
    print("Reconstructed hologram shape:", reconstructed_hologram.shape)
    print("Magnitude of a central point:", np.abs(reconstructed_hologram[ny//2, nx//2]))
    # os.system("eog recon/")
    png_files = glob.glob(os.path.join(recon_path, '*.png'))
    os.startfile(png_files[0])
