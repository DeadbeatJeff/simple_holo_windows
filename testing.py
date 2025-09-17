#!/usr/bin/env python3
# Read and transfer data: Load the digital hologram data and transfer it to the GPU as a PyTorch tensor.
import torch, cv2, numpy as np
import os
# Assuming 'hologram_data' is a NumPy array
destination_path_safe = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.join('images', 'hologram4_resized.png'))
print(destination_path_safe)
new_image=cv2.resize(destination_path_safe,(512,512))
hologram_data = cv2.imread(new_image, cv2.IMREAD_GRAYSCALE).astype(np.float32)
hologram_tensor = torch.from_numpy(hologram_data).cuda()
# Define reconstruction parameters: Set the physical constants for your optical setup, such as wavelength, pixel size, and propagation distance.
wavelength = 633e-9
pixel_size = 5e-6
distance = 0.01
# Perform Fast Fourier Transform (FFT): Use PyTorch's fft module 
# to compute the Fourier transform of the hologram. This is the
# most computationally intensive step and is where the CUDA
# acceleration provides significant speedup.
hologram_spectrum = torch.fft.fft2(hologram_tensor)
#Propagate the wavefield: Apply a propagation kernel in the frequency domain (e.g., Angular Spectrum method). This involves tensor operations that are also accelerated by CUDA.
# Define frequency space coordinates
# ...
# Calculate propagation kernel
# ...
propagated_field = hologram_spectrum * propagation_kernel
# Perform Inverse FFT: Transform the propagated wavefield back into the spatial domain.
# python
reconstructed_image = torch.fft.ifft2(propagated_field)
# Extract amplitude and phase: The reconstructed image is a complex tensor, so you will need to extract the real-valued amplitude and phase.
# amplitude = torch.abs(reconstructed_image)
# Transfer back to CPU (optional): If you need to perform CPU-based visualization or saving, move the result back to the CPU.
reconstructed_image_cpu = amplitude.cpu().numpy()

