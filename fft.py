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
    # Init processed image to padded image for now
    processed_image = padded_image

    # Based on selected mode, perform the corresponding operation
    if mode == 1:
        # Obtain 2D FFT of image
        fft_image = logic.FFT_2D(padded_image, padded_N, padded_M)
        # Crop image back to original dimensions
        fft_image = logic.crop(fft_image, N, M)
        # We only keep the magnitude of each value in the matrix
        fft_image = np.abs(fft_image)
        # Display images side by side with logarithmic scaling
        logic.display(image, fft_image, "Fourier Transform of Original Image (Log Scale)", True)
    elif mode == 2:
        # Obtain denoised image
        denoised_image = logic.denoise(padded_image, padded_N, padded_M)
        # Crop denoised image
        denoised_image = logic.crop(denoised_image, N, M)
        # Display images side by side
        logic.display(image, denoised_image, "Denoised Image", False)
    elif mode == 3:
        print("Mode 3 selected - compress")
    elif mode == 4:
        print("Mode 4 selected - plot runtime graphs")

if __name__ == "__main__":
    main()
