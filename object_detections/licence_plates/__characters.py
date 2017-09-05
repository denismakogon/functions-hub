import os
import cv2
import numpy as np
import math

from object_detections.licence_plates import __constants
from object_detections.licence_plates import __processor


class PossibleChar(object):

    def __init__(self, _contour):
        self.contour = _contour
        self.boundingRect = cv2.boundingRect(self.contour)
        (intX, intY,
         intWidth, intHeight) = self.boundingRect

        self.intBoundingRectX = intX
        self.intBoundingRectY = intY
        self.intBoundingRectWidth = intWidth
        self.intBoundingRectHeight = intHeight

        self.intBoundingRectArea = self.intBoundingRectWidth * self.intBoundingRectHeight

        self.intCenterX = (self.intBoundingRectX + self.intBoundingRectX + self.intBoundingRectWidth) / 2
        self.intCenterY = (self.intBoundingRectY + self.intBoundingRectY + self.intBoundingRectHeight) / 2

        self.fltDiagonalSize = math.sqrt((self.intBoundingRectWidth ** 2) + (self.intBoundingRectHeight ** 2))

        self.fltAspectRatio = float(self.intBoundingRectWidth) / float(self.intBoundingRectHeight)
    # end constructor

# end class


def load_knn_data_and_train():
    try:
        classifications = np.loadtxt(
            os.path.join(os.getcwd(), "licence_plates/classifications.txt"), np.float32)
        flattened_images = np.loadtxt(
            os.path.join(os.getcwd(), "licence_plates/flattened_images.txt"), np.float32)
    except Exception as ex:
        print(str(ex))
        raise ex

    reshaped = classifications.reshape((classifications.size, 1))
    __constants.K_NEAREST.setDefaultK(1)
    __constants.K_NEAREST.train(flattened_images, cv2.ml.ROW_SAMPLE, reshaped)

    return True


def detect_chars_in_plates(possible_plates):

    if len(possible_plates) == 0:
        return possible_plates

    for possible_plate in possible_plates:
        possible_plate.imgGrayscale, possible_plate.imgThresh = (
            __processor.process(possible_plate.imgPlate))

        possible_plate.imgThresh = cv2.resize(possible_plate.imgThresh, (0, 0), fx=1.6, fy=1.6)
        threshold_value, possible_plate.imgThresh = cv2.threshold(
            possible_plate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        list_of_possible_chars_in_plate = find_possible_chars_on_plate(
            possible_plate.imgGrayscale, possible_plate.imgThresh)

        list_of_lists_of_matching_chars_on_plate = find_list_of_lists_of_matching_chars(
            list_of_possible_chars_in_plate)

        for i in range(len(list_of_lists_of_matching_chars_on_plate)):
            list_of_lists_of_matching_chars_on_plate[i] = sorted(
                list_of_lists_of_matching_chars_on_plate[i],
                key=lambda matching_char: matching_char.intCenterX)
            list_of_lists_of_matching_chars_on_plate[i] = remove_inner_overlapping_chars(
                list_of_lists_of_matching_chars_on_plate[i])

        len_of_longest_list_of_chars = 0
        index_of_longest_list_of_chars = 0

        for i in range(len(list_of_lists_of_matching_chars_on_plate)):
            if len(list_of_lists_of_matching_chars_on_plate[i]) > len_of_longest_list_of_chars:
                len_of_longest_list_of_chars = len(list_of_lists_of_matching_chars_on_plate[i])
                index_of_longest_list_of_chars = i

        if len(list_of_lists_of_matching_chars_on_plate):
            longest_list_of_matching_chars_in_plate = list_of_lists_of_matching_chars_on_plate[
                index_of_longest_list_of_chars]

            possible_plate.strChars = recognize_chars_in_plate(
                possible_plate.imgThresh, longest_list_of_matching_chars_in_plate)
            pass

    return possible_plates


def find_possible_chars_on_plate(_, img_thresh):
    list_of_possible_chars = []
    img_thresh_copy = img_thresh.copy()
    img_contours, contours, hierarchy = cv2.findContours(
        img_thresh_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        possible_char = PossibleChar(contour)

        if check_if_possible_char(possible_char):
            list_of_possible_chars.append(possible_char)

    return list_of_possible_chars


def check_if_possible_char(possible_char):
    return (possible_char.intBoundingRectArea > __constants.MIN_PIXEL_AREA and
            possible_char.intBoundingRectWidth > __constants.MIN_PIXEL_WIDTH and
            possible_char.intBoundingRectHeight > __constants.MIN_PIXEL_HEIGHT and
            __constants.MIN_ASPECT_RATIO < possible_char.fltAspectRatio < __constants.MAX_ASPECT_RATIO)


def find_list_of_lists_of_matching_chars(list_of_possible_chars):
    list_of_lists_of_matching_chars = []
    for possibleChar in list_of_possible_chars:
        list_of_matching_chars = find_list_of_matching_chars(possibleChar, list_of_possible_chars)
        list_of_matching_chars.append(possibleChar)
        if len(list_of_matching_chars) < __constants.MIN_NUMBER_OF_MATCHING_CHARS:
            continue

        list_of_lists_of_matching_chars.append(list_of_matching_chars)

        list_of_possible_chars_with_current_matches_removed = (
            list(set(list_of_possible_chars) - set(list_of_matching_chars)))

        recursive = find_list_of_lists_of_matching_chars(
            list_of_possible_chars_with_current_matches_removed)

        for recursiveListOfMatchingChars in recursive:
            list_of_lists_of_matching_chars.append(recursiveListOfMatchingChars)
        break

    return list_of_lists_of_matching_chars


def find_list_of_matching_chars(possible_char, list_of_chars):
    list_of_matching_chars = []

    for possibleMatchingChar in list_of_chars:
        if possibleMatchingChar == possible_char:
            continue

        distance_between_chars = fn_distance_between_chars(possible_char, possibleMatchingChar)
        angle_between_chars = fn_angle_between_chars(possible_char, possibleMatchingChar)
        change_in_area = float(abs(possibleMatchingChar.intBoundingRectArea -
                               possible_char.intBoundingRectArea)) / float(possible_char.intBoundingRectArea)
        change_in_width = float(abs(possibleMatchingChar.intBoundingRectWidth -
                                possible_char.intBoundingRectWidth)) / float(possible_char.intBoundingRectWidth)
        change_in_height = float(abs(possibleMatchingChar.intBoundingRectHeight -
                                 possible_char.intBoundingRectHeight)) / float(possible_char.intBoundingRectHeight)

        if (distance_between_chars < (possible_char.fltDiagonalSize * __constants.MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            angle_between_chars < __constants.MAX_ANGLE_BETWEEN_CHARS and
            change_in_area < __constants.MAX_CHANGE_IN_AREA and
            change_in_width < __constants.MAX_CHANGE_IN_WIDTH and
                change_in_height < __constants.MAX_CHANGE_IN_HEIGHT):

            list_of_matching_chars.append(possibleMatchingChar)

    return list_of_matching_chars


def fn_distance_between_chars(first, second):
    return math.sqrt((abs(first.intCenterX - second.intCenterX) ** 2)
                     + (abs(first.intCenterY - second.intCenterY) ** 2))


def fn_angle_between_chars(first, second):
    adj = float(abs(first.intCenterX - second.intCenterX))
    opp = float(abs(first.intCenterY - second.intCenterY))

    angle = math.atan(opp / adj) if adj != 0.0 else 1.5708

    return angle * (180.0 / math.pi)


def remove_inner_overlapping_chars(list_of_matching_chars):
    list_of_matching_chars_with_inner_char_removed = list(list_of_matching_chars)

    for currentChar in list_of_matching_chars:
        for otherChar in list_of_matching_chars:
            if currentChar != otherChar:
                if fn_distance_between_chars(currentChar, otherChar) < (
                            currentChar.fltDiagonalSize * __constants.MIN_DIAG_SIZE_MULTIPLE_AWAY):
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:
                        if currentChar in list_of_matching_chars_with_inner_char_removed:
                            list_of_matching_chars_with_inner_char_removed.remove(currentChar)
                    else:
                        if otherChar in list_of_matching_chars_with_inner_char_removed:
                            list_of_matching_chars_with_inner_char_removed.remove(otherChar)

    return list_of_matching_chars_with_inner_char_removed


def recognize_chars_in_plate(img_thresh, list_of_matching_chars):
    str_chars = ""
    height, width = img_thresh.shape
    img_thresh_color = np.zeros((height, width, 3), np.uint8)
    list_of_matching_chars = sorted(
        list_of_matching_chars,
        key=lambda matching_char: matching_char.intCenterX)
    cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR, img_thresh_color)

    for current_char in list_of_matching_chars:
        pt1 = (current_char.intBoundingRectX, current_char.intBoundingRectY)
        pt2 = ((current_char.intBoundingRectX + current_char.intBoundingRectWidth),
               (current_char.intBoundingRectY + current_char.intBoundingRectHeight))
        cv2.rectangle(img_thresh_color, pt1, pt2, __constants.SCALAR_GREEN, 2)
        img_roi = img_thresh[
            current_char.intBoundingRectY: (current_char.intBoundingRectY +
                                            current_char.intBoundingRectHeight),
            current_char.intBoundingRectX: (current_char.intBoundingRectX +
                                            current_char.intBoundingRectWidth)
        ]

        imo_roi_resize = cv2.resize(img_roi, (__constants.RESIZED_CHAR_IMAGE_WIDTH,
                                              __constants.RESIZED_CHAR_IMAGE_HEIGHT))
        roi_resize = np.float32(imo_roi_resize.reshape(
            (1, __constants.RESIZED_CHAR_IMAGE_WIDTH * __constants.RESIZED_CHAR_IMAGE_HEIGHT)))
        _, results, neigh_resp, dists = __constants.K_NEAREST.findNearest(roi_resize, k=1)
        current_char = str(chr(int(results[0][0])))
        str_chars += current_char

    return str_chars

