import sys
import os

def validate_mode(value):
    try:
        mode = int(value)

        if mode not in [1, 2, 3, 4]:
            print("ERROR\tInvalid mode, select from [1, 2, 3, 4]")
            sys.exit(1)

        return mode

    except ValueError:
        print("ERROR\tMode must be a valid integer")
        sys.exit(1)

def validate_image_filename(value):
    if not _is_valid_file(value):
        print("ERROR\tFile name not found in directory")

    if not _is_valid_image_file(value):
        print("ERROR\tInvalid file extension detected (must be .png)")

    return value

# Private helper function to validate if provided file name exists in cwd
def _is_valid_file(filename):
    return os.path.isfile(os.path.join(os.getcwd(), filename))

# Private helper function to validate if file extension is a valid image extension
def _is_valid_image_file(filename):
    # Can add more valid extensions in the future
    valid_extensions = ['.png', '.jpg', '.jpeg']
    _, ext = os.path.splitext(filename.lower())
    return ext in valid_extensions
