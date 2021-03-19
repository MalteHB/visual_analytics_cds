#!usr/bin/env python3

from pathlib import Path

import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


def fetch_mnist():
    """Wrapper for getting the mnist dataset.

    Returns:
        X, y: Pictures and labels from the MNIST dataset.
    """

    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

    return X, y 

def setting_default_data_dir(assignment=2):
    """Setting a default data directory

    Returns:
        PosixPath: Data directory
    """

    if assignment == 2:
            
        root_dir = Path.cwd()  # Setting root directory.

        data_dir = root_dir / "data" / "17flowers"  # Setting data directory.

    if assignment == 3:

        root_dir = Path.cwd()  # Setting root directory.

        data_dir = root_dir / "data" / "ass3"  # Setting data directory.

    return data_dir


def setting_default_out_dir(assignment=2):
    """Setting a default Output directory

    Returns:
        PosixPath: Output directory
    """
    
    if assignment == 2:

        root_dir = Path.cwd()  # Setting root directory.

        out_dir = root_dir / "out"  # Setting data directory.

    return out_dir

def setting_default_target_path(assignment=2):
    """Setting a default Output directory

    Returns:
        PosixPath: Output directory
    """

    if assignment == 2:
        
        root_dir = Path.cwd()  # Setting root directory.

        target_path = root_dir / "data" / "17flowers" / "image_1360.jpg"  # Setting target path.

    if assignment == 3:
        
        root_dir = Path.cwd()  # Setting root directory.

        target_path = root_dir / "data" / "ass3" / "ass3.jpg"  # Setting target path.

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


def grab_contours(cnts):
    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
    	cnts = cnts[1]

    # otherwise OpenCV has changed their cv2.findContours return
    # signature yet again and I have no idea WTH is going on
    else:
    	raise Exception(("Contours tuple must have length 2 or 3, "
    		"otherwise OpenCV changed their cv2.findContours return "
    		"signature yet again. Refer to OpenCV's documentation "
    		"in that case"))

    # return the actual contours array
    return cnts

def translate(image, x, y):
	# Define the translation matrix and perform the translation
	M = np.float32([[1, 0, x], [0, 1, y]])
	shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

	# Return the translated image
	return shifted


def rotate(image, angle, center = None, scale = 1.0):
	# Grab the dimensions of the image
	(h, w) = image.shape[:2]

	# If the center is None, initialize it as the center of
	# the image
	if center is None:
		center = (w / 2, h / 2)

	# Perform the rotation
	M = cv2.getRotationMatrix2D(center, angle, scale)
	rotated = cv2.warpAffine(image, M, (w, h))

	# Return the rotated image
	return rotated


def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and
	# grab the image size
	dim = None
	(h, w) = image.shape[:2]

	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (int(w * r), height)

	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))

	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)

	# return the resized image
	return resized

def jimshow(image, title=False):
    """imshow with matplotlib dependencies 
    """
    # Acquire default dots per inch value of matplotlib
    dpi = mpl.rcParams['figure.dpi']

    height, width, depth = image.shape
    figsize = width / float(dpi), height / float(dpi)
    
    plt.figure(figsize=figsize)
    
    if depth == 1:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      
    if title:
        plt.title(title)
    plt.axis('off')
    
    plt.show()

def jimshow_channel(image, title=False):
    """
    Modified jimshow() to plot individual channels
    """
    # Acquire default dots per inch value of matplotlib
    dpi = mpl.rcParams['figure.dpi']

    height, width = image.shape
    figsize = width / float(dpi), height / float(dpi)
    
    plt.figure(figsize=figsize)
    
    plt.imshow(image, cmap='gray')
      
    if title:
        plt.title(title)
    plt.axis('off')
    
    plt.show()