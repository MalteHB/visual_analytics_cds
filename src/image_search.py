# Importing packages
import os

from pathlib import Path
import argparse

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

from utils.utils import setting_default_data_dir, setting_default_out_dir, setting_default_target_path, get_filepaths_from_data_dir, get_filename, load_image

def main(args):

    print("Initiating some awesome image search!")

    # Importing arguments from the arguments parser

    data_dir = args.dd

    out_dir = args.od

    target_image_filepath = args.tif

    ImageSearch(data_dir=data_dir, out_dir=out_dir, target_image_filepath=target_image_filepath)

    print(f"DONE! Have a nice day. :-)")


class ImageSearch:

    def __init__(self, data_dir=None, out_dir=None, target_image_filepath=None):

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

        self.target_image_filepath = target_image_filepath

        if self.target_image_filepath is None:

            self.target_image_filepath = setting_default_target_path()  # Setting default data directory.

            print(f"\nTarget image filepath is not specified.\nSetting it to '{self.target_image_filepath}'.\n")

        target_image = load_image(self.target_image_filepath)

        filenames = []  # Creating empty variable for filenames.

        distances = []  # Creating empty list for Chi-Squared distances.

        # For each file in the data directory, load the image, get the height, width and number of channels
        # and split the image into equally sized quadrants and save these into the output directory.
        for file in tqdm(files):

            if file != self.target_image_filepath:  # Making sure that the comparison does not happen to the target image.

                filename = get_filename(file)  # Getting filename.

                comparison_image = load_image(file)  # Loading comparison image.

                chisqr = self.get_chisqr(target_image, comparison_image)

                filenames.append(filename)  # Appending the filename.

                distances.append(chisqr)  # Appending the Chi-Squared distance measure.

        data_dict = {"filename": filenames,
                     "distance": distances}

        df = pd.DataFrame(data=data_dict)

        write_path = self.out_dir / "distances.csv"

        df.to_csv(write_path)

        print(df)


    def get_chisqr(self, image1, image2):

        hist1 = cv2.calcHist(images=[image1],  # Choosing the image.
                             channels=[0,1,2],  # Setting the channels.
                             mask=None,  # Not using a mask.
                             histSize=[8,8,8],  # Setting our histogram bins to be of size 8.
                             ranges=[0,256, 0, 256, 0, 256])  # Ranges for the possible pixel values.

        hist2 = cv2.calcHist(images=[image2],  # Choosing the image.
                             channels=[0,1,2],  # Setting the channels.
                             mask=None,  # Not using a mask.
                             histSize=[8,8,8],  # Setting our histogram bins to be of size 8.
                             ranges=[0,256, 0, 256, 0, 256])  # Ranges for the possible pixel values.

        hist1_norm = cv2.normalize(hist1,  # Parsing first histogram
                                   hist1,  # Parsing first histogram
                                   0,255,  # Setting the range of pixels
                                   cv2.NORM_MINMAX)  # Choosing the normalization 

        hist2_norm = cv2.normalize(hist2,  # Parsing second histogram
                                   hist2,  # Parsing second histogram
                                   0,255,  # Setting the range of pixels
                                   cv2.NORM_MINMAX)  # Choosing the normalization 

        chisqr = round(  # Using the round function to round to 2 decimals.
                cv2.compareHist(
                    hist1_norm,  # First normalized histogram.
                    hist2_norm,  # Second normalized histogram.
                    cv2.HISTCMP_CHISQR  # Metric for comparison
                )
        )

        return chisqr


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