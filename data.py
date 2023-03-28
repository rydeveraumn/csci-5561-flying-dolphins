# first party
import functools
import glob
import math
import os
from multiprocessing import Pool, cpu_count

# third party
import click
import cv2
import numpy as np
import pydicom
import tqdm
from cv2_rolling_ball import subtract_background_rolling_ball
from pydicom.pixel_data_handlers.util import apply_voi_lut
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.draw import polygon
from skimage.feature import canny
from skimage.filters import sobel
from skimage.transform import hough_line, hough_line_peaks


# This function performs the rolling ball algorithm on an image given the image and ball size
# Also will normalize the output image
# Recommended ball_size = 5
def rollingBall(img, ball_size):
    final_img, lightbackground = subtract_background_rolling_ball(
        img, ball_size, light_background=False, use_paraboloid=False, do_presmooth=False
    )
    image_output = cv2.normalize(
        final_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    return image_output


# Main function to find threshold
# Takes in a histogram and outputs the threshold
def Huang(data):
    threshold = -1
    first_bin = 0
    for ih in range(254):
        if data[ih] != 0:
            first_bin = ih
            break
    last_bin = 254
    for ih in range(254, -1, -1):
        if data[ih] != 0:
            last_bin = ih
            break
    term = 1.0 / (last_bin - first_bin)
    mu_0 = np.zeros(shape=(254, 1))
    num_pix = 0.0
    sum_pix = 0.0
    for ih in range(first_bin, 254):
        sum_pix = sum_pix + (ih * data[ih])
        num_pix = num_pix + data[ih]
        mu_0[ih] = sum_pix / num_pix
    min_ent = float("inf")
    for it in range(254):
        ent = 0.0
        for ih in range(it):
            mu_x = 1.0 / (1.0 + term * math.fabs(ih - mu_0[it]))
            if not ((mu_x < 1e-06) or (mu_x > 0.999999)):
                ent = ent + data[ih] * (
                    -mu_x * math.log(mu_x) - (1.0 - mu_x) * math.log(1.0 - mu_x)
                )
        if ent < min_ent:
            min_ent = ent
            threshold = it

    return threshold


# This function takes in
# an image and returns the binary version of
# it using the Huang algorithm
def getBinaryImage(image):
    histogram, bin_edges = np.histogram(image, bins=range(257))
    threshold = Huang(histogram)
    binary_image = np.where(image > threshold, 255, 0)

    return binary_image


# Takes in a binary background image and returns
# the denoised version
# Recommended values: erosion_size = (2, 4), erosion_iterations = 1, dilation_size = (2, 3), dilation_iterations = 6
def denoise(
    image, erosion_size, erosion_iterations, dilation_size, dilation_iterations
):
    # Define the structuring element for the erosion and dilation operations
    structuring_element = np.ones(erosion_size, dtype=np.uint8)

    # Perform erosion followed by dilation to remove noise and fill gaps
    # This combination seems to do well in removing the background without noise
    eroded_array = binary_erosion(image, structuring_element)
    for i in range(erosion_iterations - 1):
        eroded_array = binary_erosion(eroded_array, structuring_element)

    structuring_element = np.ones(dilation_size, dtype=np.uint8)
    denoised_array = binary_dilation(eroded_array, structuring_element)
    for i in range(dilation_iterations - 1):
        denoised_array = binary_dilation(denoised_array, structuring_element)

    return denoised_array


def removeBackground(
    img, ball_size, erosion_size, erosion_iterations, dilation_size, dilation_iterations
):
    """
    This function will take an image and use all of the previous helper functions to remove a background
    Inputs are all controllable variables in previous functions for maximum flexability
    """
    img_proc = rollingBall(img, ball_size)
    background_img = getBinaryImage(img_proc)
    background_denoised = denoise(
        background_img,
        erosion_size,
        erosion_iterations,
        dilation_size,
        dilation_iterations,
    )
    background_denoised = background_denoised.astype(np.uint8)
    img_proc = np.array(img_proc)
    img_noback = np.multiply(background_denoised, img_proc)
    return img_noback


# Orients image to right if it is left sided
def right_orient_mammogram(image):
    left_nonzero = cv2.countNonZero(image[:, 0 : int(image.shape[1] / 2)])
    right_nonzero = cv2.countNonZero(image[:, int(image.shape[1] / 2) :])

    if left_nonzero < right_nonzero:
        image = cv2.flip(image, 1)

    return image


def apply_canny(image):
    canny_img = canny(image, 6)
    return sobel(canny_img)


def get_hough_lines(canny_img):
    h, theta, d = hough_line(canny_img)
    lines = list()
    # print('\nAll hough lines')
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        # print("Angle: {:.2f}, Dist: {:.2f}".format(np.degrees(angle), dist))
        x1 = 0
        y1 = (dist - x1 * np.cos(angle)) / np.sin(angle + 0.000000001)
        x2 = canny_img.shape[1]
        y2 = (dist - x2 * np.cos(angle)) / np.sin(angle + 0.000000001)
        lines.append(
            {
                "dist": dist,
                "angle": np.degrees(angle),
                "point1": [x1, y1],
                "point2": [x2, y2],
            }
        )

    return lines


# Shortlisting lines
def shortlist_lines(lines):
    MIN_ANGLE = 10
    MAX_ANGLE = 70
    MIN_DIST = 5
    MAX_DIST = 200

    shortlisted_lines = [
        x
        for x in lines
        if (x["dist"] >= MIN_DIST)
        & (x["dist"] <= MAX_DIST)
        & (x["angle"] >= MIN_ANGLE)
        & (x["angle"] <= MAX_ANGLE)
    ]
    # print('\nShorlisted lines')
    # for i in shortlisted_lines:
    # print("Angle: {:.2f}, Dist: {:.2f}".format(i['angle'], i['dist']))

    return shortlisted_lines


def remove_pectoral_region(shortlisted_lines):
    shortlisted_lines.sort(key=lambda x: x["dist"])
    pectoral_line = shortlisted_lines[0]
    d = pectoral_line["dist"]
    theta = np.radians(pectoral_line["angle"])

    x_intercept = d / np.cos(theta)
    y_intercept = d / np.sin(theta)

    return polygon([0, 0, y_intercept], [0, x_intercept, 0])


# Only use this function, uses all of the above to return the image with the pectoral removed
def removePectoral(image):
    canny_image = apply_canny(image)
    lines = get_hough_lines(canny_image)
    shortlisted_lines = shortlist_lines(lines)
    rr, cc = remove_pectoral_region(shortlisted_lines)

    # True index
    index = np.where(
        (rr < (image.shape[0] - 1)) & (cc < (image.shape[1] - 1)), True, False
    )
    rr = rr[index]
    cc = cc[index]

    image[rr, cc] = 0
    return image


def crop_medical_image(image, output_size, threshold=20):
    """
    We will use components of opencv to remove excess space around
    medical images
    """
    # In some of our medical images there is a small frame at the outer
    # edge so we will do our best to generalize and remove this
    X = image.copy()
    X = X[5:-5, 5:-5]

    # Regions of non-empty pixels
    # Current experiemnts show that  is a good cutoff
    output = cv2.connectedComponentsWithStats(
        (X > threshold).astype(np.uint8), 8, cv2.CV_32S
    )

    # Get the stats
    stats = output[2]

    # Find the max area which always corresponds to the breast data
    index = stats[1:, 4].argmax() + 1
    x1, y1, w, h = stats[index][:4]
    x2 = x1 + w
    y2 = y1 + h

    # Cropped and cleaned image
    X_cleaned_image = X[y1:y2, x1:x2]
    X_cleaned_image = cv2.resize(X_cleaned_image, output_size)

    return X_cleaned_image


def build_preprocessed_image(
    filename, save_directory, output_size=(240, 384), read_dicom=True
):
    """
    Function that will take a file name as an input and output
    a new pre-processed image
    """
    # Read images from the dicom file
    if read_dicom:
        dicom = pydicom.dcmread(filename)
        image = apply_voi_lut(dicom.pixel_array, dicom)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_16UC1)

    else:
        image = cv2.imread(filename)

    # Crop the image around the breast and resize
    image = crop_medical_image(image, output_size=output_size, threshold=45)

    # Remove the background
    image = removeBackground(
        image,
        ball_size=5,
        erosion_size=(4, 2),
        erosion_iterations=1,
        dilation_size=(3, 2),
        dilation_iterations=6,
    )

    # I think we should right orient the image just for the
    # consistency that we mentioned
    image = right_orient_mammogram(image)

    # Remove the pectoral muscle
    try:
        image = removePectoral(image.copy())

    except IndexError:
        pass

    # Write the processed image to a new directory
    save_file_name = "_".join(filename.split("/")[-2:]).replace(".dcm", ".png")
    save_file_name = os.path.join(save_directory, save_file_name)
    cv2.imwrite(save_file_name, image)


@click.command()
@click.option("--image_directory", default="./image_data/preprocessed_data_240x384")
@click.option("--width", default=240)
@click.option("--height", default=384)
def preprocess_images_task(image_directory, width, height):
    """
    Function that will preprocess all of the images for the breast
    cancer detection
    """
    # Get all of the files
    files = glob.glob(
        "/home/rydevera3/data-science/kaggle/rsna-breast-cancer/data/train_images/*/*"
    )

    # Set up function to run in multiprocessing
    output_size = (width, height)
    image_preprocessing = functools.partial(
        build_preprocessed_image,
        save_directory=f"./image_data/{image_directory}",
        output_size=output_size,
    )

    # Will run very quick with many cores and a lot of memory
    with Pool(cpu_count() - 2) as p:
        for _ in tqdm.tqdm(p.imap_unordered(image_preprocessing, files)):
            pass


if __name__ == "__main__":
    preprocess_images_task()
