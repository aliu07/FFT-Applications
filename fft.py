import argparse
import utils
import logic
import numpy as np


def parse_input():
    parser = argparse.ArgumentParser(description='FFT Application')
    parser.add_argument('-m', type=utils.validate_mode, default=1)
    parser.add_argument('-i', type=utils.validate_image_filename, default="moonlanding.png")
    return parser.parse_args()


def main():
    # Parse user inputs
    args = parse_input()
    # Obtain mode and image inputs
    mode = args.m
    image_filename = args.i
    # Obtain 2D array representation of image
    image = logic.load_image(image_filename)
    # Get dimensions of loaded image
    N, M = image.shape
    # Pad image for FFT methods (they require dimensions to be power of 2)
    padded_image = logic.pad(image, N, M)
    # Get dimensions of padded image
    padded_N, padded_M = padded_image.shape

    # Based on selected mode, perform the corresponding operation
    if mode == 1:
        # Obtain 2D FFT of image
        fft_image = logic.FFT_2D(padded_image, padded_N, padded_M)
        # fft_image = np.fft.fft2(padded_image)
        # Crop image back to original dimensions
        fft_image = logic.crop(fft_image, N, M)
        # We only keep the magnitude of each value in the matrix
        fft_image = np.abs(fft_image)
        # Display images side by side with logarithmic scaling
        logic.display(image, fft_image, "Fourier Transform of Original Image (Log Scale)", True)
    elif mode == 2:
        # Set percentage of coefficients to keep
        percentage = 0
        # Obtain denoised image
        denoised_image = logic.denoise(padded_image, padded_N, padded_M, percentage)
        # Crop denoised image
        denoised_image = logic.crop(denoised_image, N, M)
        # Display images side by side
        logic.display(image, denoised_image, f"Denoised Image (~{percentage}% coefficients kept)", False)
    elif mode == 3:
        # Compression levels (%)
        compression_lvls = [60, 90, 93, 96, 99.9]
        # Obtain collection of compressed images
        compressed_images, compression_lvls = logic.compress(padded_image, padded_N, padded_M, compression_lvls)
        # Crop each image back to its original dimensions
        for i in range(len(compressed_images)):
            compressed_images[i] = logic.crop(compressed_images[i], N, M)
        # Display images in a 2x3 grid
        logic.display_compressed_images(image, compressed_images, [0] + compression_lvls)
    elif mode == 4:
        logic.analyze_runtime_complexity()


if __name__ == "__main__":
    main()
