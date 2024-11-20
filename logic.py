import numpy as np
import matplotlib.pyplot as plt
import cv2

from matplotlib.colors import LogNorm
from math import *



# Function to load images as a matrix of integer values
def load_image(image_filename):
    # Load the image and return it
    return np.array(cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE))



# This function pads the image matrix with zeros to make sure that its dimensions are a power of 2
# image = input image
# N = number of rows
# M = number of columns
def pad(image, N, M):
    # Find next power of 2 for both dimensions
    power_2_N = 2 ** (N-1).bit_length()
    power_2_M = 2 ** (M-1).bit_length()
    # Init padded image matrix
    padded_image = np.zeros((power_2_N, power_2_M), dtype=complex)

    # Pad each existing row with zeros
    for n in range(N):
        padded_image[n] = np.append(image[n], [0] * (power_2_M - M))

    return padded_image



# This function crops the input image to the specified dimensions.
# image = input image matrix
# N = number of rows to crop to
# M = number of columns to crop to
def crop(image, N, M):
    return image[:N, :M]



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

    # Take inverse 2D FFT of processed image
    N, M = processed_image.shape
    inverse = np.abs(inverse_FFT_2D(processed_image, N, M))

    # Display Fourier transform with logarithmic scale
    ax2.imshow(inverse, norm=LogNorm(), cmap='gray')
    ax2.set_title('Fourier Transform (Log Scale)')
    ax2.axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()



# Naive implementation of 1D DFT
# x = input vector (1D)
def DFT_1D(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)

    for k in range(N):
        sum = 0

        for n in range(N):
            sum += x[n] * np.exp((-2j * np.pi * k * n) / N)

        X[k] = sum

    return X



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
def FFT_1D(x):
    # Get length of input vector
    N = len(x)
    # While vector is not a power of 2, pad with zeros
    while not log2(N).is_integer():
        x = np.append(x, [0])
        N += 1

    # Base case: simply return x since exponential component becomes 1
    if N <= 1:
        return x

    # Split into even and odd indices
    even = FFT_1D(x[::2])
    odd = FFT_1D(x[1::2])
    # Compute factors
    factors = np.exp((-2j * np.pi * np.arange(N // 2)) / N)
    # Multiply odd indices by respective factor
    odd = np.multiply(factors, odd)
    # Init result
    result = np.zeros(N, dtype=complex)
    result[:N//2] = np.add(even, odd)
    result[N//2:] = np.subtract(even, odd)

    return result



def FFT_2D(f, N, M):
    # Init result
    result = np.zeros((N, M), dtype=complex)
    col_transformed = np.zeros((N, M), dtype=complex)

    # Apply 1D FFT to each column
    for i in range(M):
        col_transformed[:, i] = FFT_1D(f[:, i])

    # Apply 1D FFT to each row
    for j in range(N):
        result[j] = FFT_1D(col_transformed[j])

    return result

# This function takes the inverse FFT of a 1D vector X.
# X = input vector
def inverse_FFT_1D(X):
    N = len(X)
    # Base case
    if N <= 1:
        return X

    # Split and recurse
    even = inverse_FFT_1D(X[::2])
    odd = inverse_FFT_1D(X[1::2])

    # Compute twiddle factors with correct IFFT sign
    factors = np.exp(-2j * np.pi * np.arange(N) / N)

    # Combine results with normalization
    result = np.zeros(N, dtype=complex)
    for k in range(N//2):
        result[k] = (even[k] + factors[k] * odd[k]) / 2
        result[k + N//2] = (even[k] - factors[k] * odd[k]) / 2

    return result



def inverse_FFT_2D(F, N, M):
    # Find next power of 2 for both dimensions
    power_2_N = 2 ** (N-1).bit_length()
    power_2_M = 2 ** (M-1).bit_length()
    # Pad input with zeros
    padded_F = np.pad(F, ((0, power_2_N - N), (0, power_2_M - M)), mode='constant', constant_values=0)
    # Init result
    result = np.zeros((power_2_N, power_2_M), dtype=complex)

    # Apply 1D inverse FFT to rows
    row_transformed = np.zeros((power_2_N, power_2_M), dtype=complex)

    for i in range(power_2_N):
        row_transformed[i] = inverse_FFT_1D(padded_F[i])

    # Apply 1D inverse FFT to cols
    for j in range(power_2_M):
        result[:, j] = inverse_FFT_1D(row_transformed[:, j])

    # Crop back to original dimensions
    result = result[:N, :M]

    return result



# This function denoises an image by removing frequencies in the Fourier domain
# bounded by the provided lower and upper bounds
# f = input 2D matrix of image
# N = number of rows
# M = number of columns
# lower = lower bound value
# upper = upper bound value
def denoise(f, N, M, lower=np.pi, upper=1.5 * np.pi):
    # Compute FFT of image
    F = FFT_2D(f, N, M)
    # Calculate phase of each value in image
    phase = np.angle(F)
    # Calculate fundamental frequencies by taking the modulo by 2pi
    fundamental_freq = np.mod(phase, 2 * np.pi)
    # Filter out values by creating a mask
    mask = ~((fundamental_freq >= lower) & (fundamental_freq <= upper))
    F_filtered = F * mask
    # Print statistics about how many frequencies were filtered
    total_coeffs = N * M
    kept_coeffs = np.count_nonzero(mask)
    print(f"Kept {kept_coeffs} out of {total_coeffs} coefficients ({kept_coeffs/total_coeffs:.2%})")

    return F_filtered
