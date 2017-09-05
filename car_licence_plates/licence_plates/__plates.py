import cv2
import math

from car_licence_plates.licence_plates import __characters
from car_licence_plates.licence_plates import __constants
from car_licence_plates.licence_plates import __processor


class PossiblePlate(object):
    def __init__(self):
        self.imgPlate = None
        self.imgGrayscale = None
        self.imgThresh = None

        self.rr_location_of_plate_in_scene = None

        self.strChars = ""


def detect_plates_in_scene(img_original_scene):
    list_of_possible_plates = []

    cv2.destroyAllWindows()

    _, img_thresh_scene = __processor.process(img_original_scene)

    list_of_possible_chars_in_scene = find_possible_chars_in_scene(img_thresh_scene)

    list_of_lists_of_matching_chars_in_scene = __characters.find_list_of_lists_of_matching_chars(
        list_of_possible_chars_in_scene)

    for list_of_matching_chars in list_of_lists_of_matching_chars_in_scene:
        possible_plate = extract_plate(img_original_scene, list_of_matching_chars)

        if possible_plate.imgPlate is not None:
            list_of_possible_plates.append(possible_plate)

    return list_of_possible_plates


def find_possible_chars_in_scene(img_thresh):
    list_of_possible_chars = []
    img_thresh_copy = img_thresh.copy()
    img_contours, contours, hierarchy = cv2.findContours(
        img_thresh_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):

        possible_char = __characters.PossibleChar(contours[i])

        if __characters.check_if_possible_char(possible_char):
            list_of_possible_chars.append(possible_char)

    return list_of_possible_chars


def extract_plate(img_original, list_of_matching_chars):
    possible_plate = PossiblePlate()

    list_of_matching_chars = sorted(
        list_of_matching_chars, key=lambda matching_char: matching_char.intCenterX)

    x = (list_of_matching_chars[0].intCenterX +
         list_of_matching_chars[len(list_of_matching_chars) - 1].intCenterX) / 2.0
    y = (list_of_matching_chars[0].intCenterY +
         list_of_matching_chars[len(list_of_matching_chars) - 1].intCenterY) / 2.0

    center = x, y

    width_p = int(
        (list_of_matching_chars[
             len(list_of_matching_chars) - 1
         ].intBoundingRectX +
         list_of_matching_chars[len(list_of_matching_chars) - 1].intBoundingRectWidth -
         list_of_matching_chars[0].intBoundingRectX) * __constants.PLATE_WIDTH_PADDING_FACTOR)

    total_of_char_heights = 0

    for matchingChar in list_of_matching_chars:
        total_of_char_heights += matchingChar.intBoundingRectHeight

    height_p = (int(total_of_char_heights / len(list_of_matching_chars) *
                    __constants.PLATE_HEIGHT_PADDING_FACTOR))

    opposite = list_of_matching_chars[
                      len(list_of_matching_chars) - 1].intCenterY - list_of_matching_chars[0].intCenterY
    hypotenuse = __characters.fn_distance_between_chars(
        list_of_matching_chars[0], list_of_matching_chars[len(list_of_matching_chars) - 1])
    correction_angle_in_deg = math.asin(opposite / hypotenuse) * (180.0 / math.pi)

    possible_plate.rr_location_of_plate_in_scene = (
        center, (width_p, height_p), correction_angle_in_deg)

    rotation_matrix = cv2.getRotationMatrix2D(center, correction_angle_in_deg, 1.0)

    height, width, _ = img_original.shape

    rotated = cv2.warpAffine(img_original, rotation_matrix, (width, height))

    cropped = cv2.getRectSubPix(rotated, (width_p, height_p), center)

    possible_plate.imgPlate = cropped

    return possible_plate
