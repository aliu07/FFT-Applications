import argparse
import utils

def parse_input():
    parser = argparse.ArgumentParser(description='FFT Application')
    parser.add_argument('-m', type=utils.validate_mode, default=1)
    parser.add_argument('-i', type=utils.validate_image_filename, default="moonlanding.png")
    return parser.parse_args()

def main():
    # Parse user inputs
    args = parse_input()
    # Obtain mode and image
    mode = args.m
    image = args.i

if __name__ == "__main__":
    main()
