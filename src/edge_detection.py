# Using the skills you have learned up to now, do the following tasks:



# Draw a green rectangular box to show a region of interest (ROI) around the main body of text in the middle of the image. Save this as image_with_ROI.jpg.
# Crop the original image to create a new image containing only the ROI in the rectangle. Save this as image_cropped.jpg.
# Using this cropped image, use Canny edge detection to 'find' every letter in the image
# Draw a green contour around each letter in the cropped image. Save this as image_letters.jpg


# TIPS

# Remember all of the skills you've learned so far and think about how they might be useful
# This means: colour models; cropping; masking; simple and adaptive thresholds; binerization; mean, median, and Gaussian blur.
# Experiment with different approaches until you are able to find as many of the letters and punctuation as possible with the least amount of noise. You might not be able to remove all artifacts - that's okay!


# Bonus challenges

# If you want to push yourself, try to write a script which runs from the command line and which takes any similar input (an image containing text) and produce a similar output (a new image with contours drawn around every letter).


# Importing packages

from pathlib import Path
import argparse

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

from utils.utils import setting_default_data_dir
from utils.utils import setting_default_out_dir
from utils.utils import setting_default_target_path
from utils.utils import get_filepaths_from_data_dir 
from utils.utils import get_filename
from utils.utils import load_image
from utils.utils import jimshow, jimshow_channel

def main(args):

    print("Initiating some awesome image search!")

    # Importing arguments from the arguments parser

    data_dir = args.dd

    out_dir = args.od

    target_image_filepath = args.tif

    out_path_ROI = None#args.opROI

    edge_detection = EdgeDetection(data_dir=data_dir, 
                                   out_dir=out_dir)

    edge_detection.create_image_ROI(target_image_filepath=target_image_filepath,
                                    out_path=out_path_ROI,
                                    pt1=(2900,2800),
                                    pt2=(1400,875),
                                    color=(0,255,0),
                                    thickness=3)

    edge_detection.crop_image(target_image_filepath=target_image_filepath,
                                    out_path=out_path_ROI,
                                    pt1=(2900,2800),
                                    pt2=(1400,875),
                                    color=(255),
                                    thickness=-1)



    print(f"DONE! Have a nice day. :-)")


class EdgeDetection:

    def __init__(self, data_dir=None, out_dir=None, target_image_filepath=None):

        self.data_dir = data_dir

        if self.data_dir is None:

            self.data_dir = setting_default_data_dir(assignment=3)  # Setting default data directory.

            print(f"\nData directory is not specified.\nSetting it to '{self.data_dir}'.")

        self.out_dir = out_dir

        if self.out_dir is None:

            self.out_dir = setting_default_out_dir(assignment=2)  # Setting default output directory.

            print(f"\nOutput directory is not specified.\nSetting it to '{self.out_dir}'.")

        self.out_dir.mkdir(parents=True, exist_ok=True)  # Making sure output directory exists.

        files = get_filepaths_from_data_dir(self.data_dir)  # Getting all the absolute filepaths from the data directory.

    def create_image_ROI(self, target_image_filepath, out_path, pt1=(2900,2800), pt2=(1400,875), color=(0,255,0), thickness=3):

        if target_image_filepath is None:

            target_image_filepath = setting_default_target_path(assignment=3)  # Setting default data directory.

            print(f"\nTarget image filepath is not specified.\nSetting it to '{target_image_filepath}'.\n")

        if out_path is None:

            out_path = self.out_dir / "image_with_ROI.jpg"

            print(f"\Output image ROI filepath is not specified.\nSetting it to '{out_path}'.\n")

        target_image = load_image(target_image_filepath)

        cv2.rectangle(target_image, pt1=pt1, pt2=pt2, color=color, thickness=thickness)

        cv2.imwrite(str(out_path), target_image)

    def crop_image(self, target_image_filepath, out_path, pt1=(2900,2800), pt2=(1400,875), color=(255), thickness=-1):

        if target_image_filepath is None:

            target_image_filepath = setting_default_target_path(assignment=3)  # Setting default data directory.

            print(f"\nTarget image filepath is not specified.\nSetting it to '{target_image_filepath}'.\n")

        if out_path is None:

            out_path = self.out_dir / "image_cropped.jpg"

            print(f"\Output image ROI filepath is not specified.\nSetting it to '{out_path}'.\n")

        target_image = load_image(target_image_filepath)

        mask = np.zeros(target_image.shape[:2], dtype="uint8")

        cv2.rectangle(mask, pt1=pt1, pt2=pt2, color=color, thickness=thickness)

        cropped = cv2.bitwise_and(target_image, target_image, mask=mask)

        cv2.imwrite(str(out_path), cropped)


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
                        metavar="target_image_filepath",
                        type=str,
                        help='Path of the file of the target image',
                        required=False)           

    main(parser.parse_args())