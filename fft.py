import argparse
import utils
import logic

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
    # Init processed image to original image for now
    processed_image = image

    # Based on selected mode, perform the corresponding operation
    if mode == 1:
        processed_image = logic.FFT_2D(image, N, M)
    elif mode == 2:
        processed_image = logic.denoise(image, N, M)
    elif mode == 3:
        print("Mode 3 selected - compress")
    elif mode == 4:
        print("Mode 4 selected - plot runtime graphs")

    # Display images side by side
    logic.display(image, processed_image)

if __name__ == "__main__":
    main()
