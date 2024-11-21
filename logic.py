import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
import math

from matplotlib.colors import LogNorm


# Function to load images as a matrix of integer values
def load_image(image_filename):
    # Load the image and return it
    return np.array(cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE))


# This function pads the image matrix with zeros to make sure that its
# dimensions are a power of 2
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
def display(original_image, processed_image, processed_image_title, isLogScale):
    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Display original image
    ax1.imshow(original_image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Display Fourier transform with logarithmic scale if need be
    if isLogScale:
        ax2.imshow(processed_image, norm=LogNorm(), cmap='gray')
    else:
        ax2.imshow(processed_image, cmap='gray')
    ax2.set_title(processed_image_title)
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
    factors = np.exp(2j * np.pi * np.arange(N // 2) / N)

    # Multiply odd indices by respective factor
    odd = np.multiply(factors, odd)

    # Combine results with normalization
    result = np.zeros(N, dtype=complex)
    result[:N//2] = np.add(even, odd)
    result[N//2:] = np.subtract(even, odd)

    return result


def inverse_FFT_2D(F, N, M):
    # Init result
    result = np.zeros((N, M), dtype=complex)
    col_transformed = np.zeros((N, M), dtype=complex)

    # Apply inverse 1D FFT to each column
    for i in range(M):
        col_transformed[:, i] = inverse_FFT_1D(F[:, i])

    # Apply inverse 1D FFT to each row
    for j in range(N):
        result[j] = inverse_FFT_1D(col_transformed[j])

    return result


# This function denoises an image by removing frequencies in the Fourier domain
# bounded by the provided lower and upper bounds
# f = input 2D matrix of image
# N = number of rows
# M = number of columns
# percentage = percentage of coefficients to keep (50% by default)
def denoise(f, N, M, percentage=50):
    # Compute FFT of image
    F = FFT_2D(f, N, M)

    # Shift zero frequency components to center
    F_shifted = np.fft.fftshift(F)

    # Create mask for specified percentage
    center_row, center_col = N//2, M//2

    # Calculate radius that keeps the percentage specified of coefficients
    total_area = N * M
    desired_area = total_area * percentage / 100
    radius = int(math.sqrt(desired_area / math.pi))

    # Create circular mask
    mask = np.zeros((N, M))
    y, x = np.ogrid[-center_row:N-center_row, -center_col:M-center_col]
    mask_area = x*x + y*y <= radius*radius
    mask[mask_area] = 1

    # Apply mask and count non-zero coefficients
    F_filtered = F_shifted * mask
    non_zeros = np.count_nonzero(F_filtered)
    total_coeffs = N * M

    print(f"Number of non-zero coefficients: {non_zeros}")
    print(f"Fraction of original coefficients: {non_zeros/total_coeffs:.2%}")

    # Shift back
    F_filtered = np.fft.ifftshift(F_filtered)

    # Take inverse FFT
    filtered_image = np.real(inverse_FFT_2D(F_filtered, N, M))

    return filtered_image
