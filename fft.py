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

    # Based on selected mode, perform the corresponding operation
    if mode == 1:
        processed_image = logic.FFT_2D(image, image.shape[0], image.shape[1])
        logic.display(image, processed_image)
    elif mode == 2:
        print("Mode 2 selected - denoise")
    elif mode == 3:
        print("Mode 3 selected - compress")
    elif mode == 4:
        print("Mode 4 selected - plot runtime graphs")

if __name__ == "__main__":
    main()
