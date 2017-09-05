import cv2

from . import __constants


def process(img_original):
    grayscale = extract_value(img_original)
    max_contrast_grayscale = maximize_contrast(grayscale)
    blurred = cv2.GaussianBlur(
        max_contrast_grayscale,
        __constants.GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, __constants.ADAPTIVE_THRESH_BLOCK_SIZE,
        __constants.ADAPTIVE_THRESH_WEIGHT)

    return grayscale, thresh


def extract_value(img_original):
    _, _, img_value = cv2.split(cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV))
    return img_value


def maximize_contrast(img_grayscale):
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    return cv2.subtract(
        cv2.add(img_grayscale, cv2.morphologyEx(img_grayscale,
                                                cv2.MORPH_TOPHAT,
                                                structuring_element)),
        cv2.morphologyEx(img_grayscale, cv2.MORPH_BLACKHAT, structuring_element)
    )
