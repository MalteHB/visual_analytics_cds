#!usr/bin/env python3

from pathlib import Path

import cv2


def setting_default_data_dir():
    """Setting a default data directory

    Returns:
        PosixPath: Data directory
    """

    root_dir = Path.cwd()  # Setting root directory.

    data_dir = root_dir / "data" / "17flowers"  # Setting data directory.

    return data_dir


def setting_default_out_dir():
    """Setting a default Output directory

    Returns:
        PosixPath: Output directory
    """
    root_dir = Path.cwd()  # Setting root directory.

    out_dir = root_dir / "out"  # Setting data directory.

    return out_dir

def setting_default_target_path():
    """Setting a default Output directory

    Returns:
        PosixPath: Output directory
    """
    root_dir = Path.cwd()  # Setting root directory.

    target_path = root_dir / "data" / "17flowers" / "image_1360.jpg"  # Setting target path.

    return target_path


def get_filepaths_from_data_dir(data_dir, file_extension="*.jpg"):
    """Creates a list containing paths to filenames in a data directoryl

    Args:
        data_dir (PosixPath): PosixPath to the data directory.
        file_extension (str): A string with the given file extension you want to extract.
    """

    files = [file for file in data_dir.glob(file_extension) if file.is_file()]  # Using list comprehension to get all the file names if they are files.

    return files


def get_filename(file):
    """Creates a list of filenames in a directory.

    Args:
        files (list): List of file paths

    Returns:
        filename: list of filenames
    """

    filename = file.name  # I take the last snippet of the path which is the file and the file extension.

    return filename


def load_text(file):
    """Loads an image.

    Args:
        file (PosixPath): A path to an image file.

    Returns:
        numpy.ndarray: NumPy Array containg all the pixels for the image.
    """

    # Read each file.

    with open(file, encoding="utf-8") as f:

        try:

            text = f.read()

        except TypeError:

            print("wtf")

        f.close()

    return text

def load_image(file):
        """Loads an image.

        Args:
            file (PosixPath): A path to an image file.

        Returns:
            numpy.ndarray: NumPy Array containg all the pixels for the image.
        """

        image = cv2.imread(str(file))

        return image
