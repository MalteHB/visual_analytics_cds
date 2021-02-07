# Importing packages
import os

from pathlib import Path
import argparse

import numpy as np
import cv2

def main(args):

    print("Initiating some awesome basic image processing!")

    # Importing arguments from the arguments parser

    data_dir = args.data_dir

    out_dir = args.out_dir

    SplittingPicturesIntoQuadrants(data_dir=data_dir, out_dir=out_dir)

    print(f"DONE! Have a nice day. :-)")



class SplittingPicturesIntoQuadrants:

    def __init__(self, data_dir=None, out_dir=None):

        self.data_dir = data_dir

        self.out_dir = out_dir

        if self.data_dir is None:

            self.data_dir = self.setting_default_data_dir()  # Setting default data directory.

        if self.out_dir is None:

            self.out_dir = self.setting_default_out_dir()  # Setting default output directory.

        self.out_dir.mkdir(parents=True, exist_ok=True)  # Making sure output directory exists.

        files = self.get_filepaths_from_data_dir(self.data_dir)  # Getting all the absolute filepaths from the data directory.

        # For each file in the data directory, load the image, get the height, width and number of channels
        # and split the image into equally sized quadrants and save these into the output directory.
        for file in files:

            filename = self.get_filename(file)

            image = self.load_image(file)

            out_path = self.out_dir / filename

            height, width, channels = self.get_width_height_and_n_channel(image)

            self.split_and_save_image(image=image, 
                                      out_path=out_path, 
                                      height=height, 
                                      width=width, 
                                      channels=channels)

    def setting_default_data_dir(self):
        """Setting a default data directory

        Returns:
            PosixPath: Data directory
        """

        root_dir = Path.cwd()  # Setting root directory.

        data_dir = root_dir / "data" / "makeup"  # Setting data directory.

        return data_dir

    def setting_default_out_dir(self):
        """Setting a default Output directory

        Returns:
            PosixPath: Output directory
        """
        root_dir = Path.cwd()  # Setting root directory.

        data_dir = root_dir / "data" / "makeup_splits"  # Setting data directory.

        return data_dir

    def get_filepaths_from_data_dir(self, data_dir, file_extension="*.jpeg"):
        """Creates a list containing paths to filenames in a data directoryl

        Args:
            data_dir (PosixPath): PosixPath to the data directory.
            file_extension (str): A string with the given file extension you want to extract.
        """

        files = [file for file in data_dir.glob(file_extension) if file.is_file()]  # Using list comprehension to get all the file names if they are files.

        return files

    def get_filename(self, file):
        """Creates a list of filenames in a directory.

        Args:
            files (list): List of file paths

        Returns:
            filename: list of filenames
        """

        filename = os.path.split(file)[-1]  # I take the last snippet of the path which is the file and the file extension.

        return filename

    def load_image(self, file):
        """Loads an image.

        Args:
            file (PosixPath): A path to an image file.

        Returns:
            numpy.ndarray: NumPy Array containg all the pixels for the image.
        """

        image = cv2.imread(str(file))

        return image



    def get_width_height_and_n_channel(self, image):
        """Gets the shape of an image.

        Args:
            image (numpy.ndarray): A NumPy array containing the BGR pixel values.

        Returns:
            int: The height of an image.
            int: The width of an image.
            int: The number of channels of an image.
        """

        height, width, channels = image.shape[0], image.shape[1], image.shape[2]

        return height, width, channels

    def split_and_save_image(self, image, out_path, height, width):
        """Splits an image into four equally sized quadrants and saves it into an output directory.

        Args:
            image (numpy.ndarray): NumPy Array containg all the pixels for the image
            out_path (PosixPath): Output path for the image.
            int: The height of the image.
            int: The width of the image.
        """
        

        filename = os.path.split(out_path)[1]  # Gets the filename

        out_path = os.path.split(out_path)[0] + "/"  # Appends a slash to the output directory

        filename_no_extention = os.path.splitext(filename)[0]  # Removal of file extenton
     
        middle_height = height//2  # Half the height in integer

        middle_width = width//2  # Half the width in integer

        upper_left = image[0:middle_height, 0:middle_width]

        upper_right = image[0:middle_height, middle_width:width]

        lower_right = image[middle_height:height, middle_width:width]
        
        lower_left = image[middle_height:height, 0:middle_width]

        cv2.imwrite(f"{out_path}{filename_no_extention}_upper_left_{str(middle_width)}x{str(middle_height)}.jpg", upper_left)

        cv2.imwrite(f"{out_path}{filename_no_extention}_upper_right_{str(middle_width)}x{str(middle_height)}.jpg", upper_right)

        cv2.imwrite(f"{out_path}{filename_no_extention}_lower_right_{str(middle_width)}x{str(middle_height)}.jpg", lower_right)

        cv2.imwrite(f"{out_path}{filename_no_extention}_lower_left_{str(middle_width)}x{str(middle_height)}.jpg", lower_left)
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',
                        metavar="data_dir",
                        type=str,
                        help='A PosixPath to the data directory',
                        required=False)

    parser.add_argument('--out_dir',
                        metavar="out_dir",
                        type=str,
                        help='A path to the output directory',
                        required=False)                

    main(parser.parse_args())