import cv2
import math
import sys

from car_licence_plates.licence_plates import __constants
from car_licence_plates.licence_plates import __characters
from car_licence_plates.licence_plates import __plates


def detect(frame):
    __characters.load_knn_data_and_train()
    if frame is None:
        print("\nerror: image not read from file \n\n")
        sys.exit(1)

    list_of_plates = __plates.detect_plates_in_scene(frame)
    list_of_possible_plates = __characters.detect_chars_in_plates(list_of_plates)
    if len(list_of_possible_plates) == 0:
        print("no license plates were detected")
        sys.exit(1)
    else:
        list_of_possible_plates = sorted(
            list_of_possible_plates,
            key=lambda possible_plate: len(possible_plate.strChars),
            reverse=True)
        licence_plate = list_of_possible_plates[0]
        if len(licence_plate.strChars) == 0:
            print("no characters were detected")
            sys.exit(1)
        __draw_red_rectangle_around_plate(frame, licence_plate)
        return licence_plate.strChars, frame


def __draw_red_rectangle_around_plate(img, licence_plate):
    points = cv2.boxPoints(licence_plate.rr_location_of_plate_in_scene)
    for i in range(4):
        cv2.line(img, tuple(points[i]),
                 tuple(points[int(math.fmod(i + 1, 4))]),
                 __constants.SCALAR_RED, 2)
