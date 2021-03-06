
import argparse

import cv2

from utils.utils import setting_default_data_dir
from utils.utils import setting_default_out_dir
from utils.utils import setting_default_target_path
from utils.utils import load_image


def main(args):

    print("Initiating some awesome image search!")

    # Importing arguments from the arguments parser

    data_dir = args.dd

    out_dir = args.od

    target_image_filepath = args.tif

    out_path_ROI = args.opROI

    out_path_letters = args.opl

    edge_detection = EdgeDetection(data_dir=data_dir,
                                   out_dir=out_dir)

    edge_detection.create_image_ROI(target_image_filepath=target_image_filepath,
                                    out_path=out_path_ROI,
                                    pt1=(2900, 2800),
                                    pt2=(1400, 875),
                                    color=(0, 255, 0),
                                    thickness=3)

    edge_detection.crop_image(target_image_filepath=target_image_filepath,
                              out_path=out_path_ROI,
                              end_point=(2900, 2800),
                              start_point=(1400, 875))

    edge_detection.find_letters(target_image_filepath=target_image_filepath,
                                out_path=out_path_letters)

    print("DONE! Have a nice day. :-)")


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

    def create_image_ROI(self, target_image_filepath, out_path, pt1=(2900, 2800), pt2=(1400, 875), color=(0, 255, 0), thickness=3):

        if target_image_filepath is None:

            target_image_filepath = setting_default_target_path(assignment=3)  # Setting default data directory.

            print(f"\nTarget image filepath is not specified.\nSetting it to '{target_image_filepath}'.\n")

        if out_path is None:

            out_path = self.out_dir / "image_with_ROI.jpg"

            print(f"\nOutput image ROI filepath is not specified.\nSetting it to '{out_path}'.\n")

        target_image = load_image(target_image_filepath)

        cv2.rectangle(target_image, pt1=pt1, pt2=pt2, color=color, thickness=thickness)

        cv2.imwrite(str(out_path), target_image)

    def crop_image(self, target_image_filepath, out_path, start_point=(1400, 875), end_point=(2900, 2800)):

        if target_image_filepath is None:

            target_image_filepath = setting_default_target_path(assignment=3)  # Setting default data directory.

            print("\nTarget image filepath is not specified.\n),",
                  f"Setting it to '{target_image_filepath}'.\n")

        if out_path is None:

            out_path = self.out_dir / "image_cropped.jpg"

            print("\nOutput image ROI filepath is not specified.",
                  f"\nSetting it to '{out_path}'.\n")

        target_image = load_image(target_image_filepath)

        # mask = np.zeros(target_image.shape[:2], dtype="uint8")

        # cv2.rectangle(mask, pt1=pt1, pt2=pt2, color=color, thickness=thickness)

        # cropped = cv2.bitwise_and(target_image, target_image, mask=mask)

        cropped = target_image[start_point[1]:end_point[1], start_point[0]:end_point[0]]

        cv2.imwrite(str(out_path), cropped)

    def find_letters(self, target_image_filepath, out_path, ):

        if target_image_filepath is None:

            target_image_filepath = self.out_dir / "image_cropped.jpg"  # Setting default data directory.

            print(f"\nTarget image filepath is not specified.\nSetting it to '{target_image_filepath}'.\n")

        if out_path is None:

            out_path = self.out_dir / "image_letters.jpg"

            print(f"\nOutput image ROI filepath is not specified.\nSetting it to '{out_path}'.\n")

        target_image = load_image(target_image_filepath)

        grey_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(grey_image, (5, 5), 0)

        canny = cv2.Canny(blurred, 90, 150)

        (contours, _) = cv2.findContours(canny,
                                         cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)

        letters = cv2.drawContours(target_image,  # Draw contours on original
                                   contours,    # Our list of contours
                                   -1,            # Which contours to draw
                                   (0, 255, 0),   # Contour color
                                   2)             # Contour pixel width

        cv2.imwrite(str(out_path), letters)


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

    parser.add_argument('--opROI',
                        metavar="Output Path Region of Interest",
                        type=str,
                        help='Output path of the image with the green ROI.',
                        required=False)

    parser.add_argument('--opl',
                        metavar="Output Path Letters",
                        type=str,
                        help='Output path of the image with letter recognition.',
                        required=False)

    main(parser.parse_args())
