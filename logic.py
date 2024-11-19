import numpy as np
import matplotlib.pyplot as plt
import cv2

from matplotlib.colors import LogNorm

# Function to load images as a matrix of integer values
def load_image(image_filename):
    # Load the image and return it
    return np.array(cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE))



# Naive implementation of 1D DFT
# x = input vector (1D)
# N = length of x
def DFT_1D(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    factors = np.exp((-2j * np.pi * k * n) / N)
    return np.dot(factors, x)



# Given a 2D array of discrete values, perform the naive 2D discrete Fourier transform
# f = input 2D matrix
# N = number of rows
# M = number of columns
def DFT_2D(f, N, M):
    # Init result
    result = np.zeros((N, M), dtype=complex)

    # Apply 1D DFT to each row
    row_transformed = np.zeros((N, M), dtype=complex)

    for i in range(N):
        row_transformed[i] = DFT_1D(f[i])

    # Apply 1D FFT to each column of the transformed rows
    for j in range(M):
        result[:, j] = DFT_1D(row_transformed[:, j])

    return result



# This function uses the Cooley-Tukey algorithm to compute the 1D FFT.
# It uses a divide and conquer approach to improve performance (a lot faster than naive implementation)
# x = input vector
# N = length of vector x
def FFT_1D(x):
    # Get length of input vector
    N = len(x)

    # Base case
    if N <= 1:
        return x

    # Split array into even and odd indices
    even = x[::2]
    odd = x[1::2]

    # Recursive calls -> divide and conquer part
    even_transformed = FFT_1D(even)
    odd_transformed = FFT_1D(odd)

    # Firstly, compute the factor for each index
    factors = np.exp((-2j * np.pi * np.arange(N // 2)) / N)
    # We then combine even and odd results
    result = np.zeros(N, dtype=complex)
    result[:N//2] = even_transformed + factors * odd_transformed
    result[N//2:] = even_transformed - factors * odd_transformed

    return result



# This function uses the Cooley-Tukey algorithm to compute the 2D DFT of the given image f.
# f = input 2D matrix
# N = number of rows
# M = number of columns
def FFT_2D(f, N, M):
    # Find next power of 2 for both dimensions
    power_2_N = 2 ** (N-1).bit_length()
    power_2_M = 2 ** (M-1).bit_length()

    # Pad input with zeros to reach power of 2 dimensions
    padded_f = np.pad(f, ((0, power_2_N - N), (0, power_2_M - M)), mode='constant', constant_values=0)
    # Init result
    result = np.zeros((power_2_N, power_2_M), dtype=complex)

    # Apply 1D FFT to each row in the matrix
    row_transformed = np.zeros((power_2_N, power_2_M), dtype=complex)

    for i in range(N):
        row_transformed[i] = FFT_1D(padded_f[i])

    # Apply 1D FFT to each column of the transformed rows
    for j in range(M):
        result[:, j] = FFT_1D(row_transformed[:, j])

    # Crop result back to original size
    result = result[:N, :M]

    return result



# This function intakes 2 images and displays them side by side
# The first param is the original image, the second param is the
# processed image
def display(original_image, processed_image):
    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Display original image
    ax1.imshow(original_image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Compute magnitude spectrum and shift zero frequency to center
    magnitude_spectrum = np.abs(processed_image)
    shifted_magnitude_spectrum = np.fft.fftshift(magnitude_spectrum)

    # Display Fourier transform with logarithmic scale
    ax2.imshow(shifted_magnitude_spectrum, norm=LogNorm(), cmap='gray')
    ax2.set_title('Fourier Transform (Log Scale)')
    ax2.axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
