# Importing packages
import os

from pathlib import Path
import argparse

import numpy as np
import cv2

from utils.utils import setting_default_data_dir, setting_default_out_dir, get_filepaths_from_data_dir, get_filename, load_image

def main(args):

    print("Initiating some awesome image search!")

    # Importing arguments from the arguments parser

    data_dir = args.data_dir

    out_dir = args.out_dir

    target_image_path = args.target_image_path

    ImageSearch(data_dir=data_dir, out_dir=out_dir)

    print(f"DONE! Have a nice day. :-)")


class ImageSearch:

    def __init__(self, data_dir=None, out_dir=None):

        self.data_dir = data_dir

        if self.data_dir is None:

            self.data_dir = setting_default_data_dir()  # Setting default data directory.

            print(f"\nData directory is not specified.\nSetting it to '{self.data_dir}'.")

        self.out_dir = out_dir

        if self.out_dir is None:

            self.out_dir = setting_default_out_dir()  # Setting default output directory.

            print(f"\nOutput directory is not specified.\nSetting it to '{self.out_dir}'.")

        self.out_dir.mkdir(parents=True, exist_ok=True)  # Making sure output directory exists.

        files = get_filepaths_from_data_dir(self.data_dir)  # Getting all the absolute filepaths from the data directory.

        target_image

        # For each file in the data directory, load the image, get the height, width and number of channels
        # and split the image into equally sized quadrants and save these into the output directory.
        for file in files:

            filename = get_filename(file)

            image = load_image(file)

            out_path = self.out_dir / filename

            

            hist1 = cv2.calcHist(images=[image], channels=[0,1,2], mask=None, histSize=[8,8,8], ranges=[0,256, 0, 256, 0, 256])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dd',
                        metavar="Data Directory",
                        type=str,
                        help='A PosixPath to the data directory.',
                        required=False)

    parser.add_argument('--od',
                        metavar="Output Directory",
                        type=str,
                        help='A path to the output directory.',
                        required=False)               

    parser.add_argument('--tif',
                        metavar="target_image_filename",
                        type=str,
                        help='Name of the file of the target image',
                        required=False)           

    main(parser.parse_args())