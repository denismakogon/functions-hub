import cv2

from object_detections.licence_plates import detect as licence_detect
from object_detections.smiles import detect as smile_detect

if __name__ == "__main__":

    plate_path = ("/Users/denismakogon/Documents/oracle"
                  "/go/src/github.com/denismakogon/functions-hub/"
                  "car_licence_plates/images/licence_plates/1.png")
    plate_text, frame = licence_detect.detect(cv2.imread(plate_path))
    print("Licence plate: ", plate_text)
    cv2.imshow("licence plate detection", frame)
    cv2.waitKey()

    smile_path = ("/Users/denismakogon/Documents/oracle/go/"
                  "src/github.com/denismakogon/functions-hub/"
                  "car_licence_plates/images/smiles/4.jpg")

    has_smile, frame = smile_detect.detect_face(cv2.imread(smile_path))
    print("Has smile? ", "Yes" if has_smile else "No")
    cv2.imshow("smile detection", frame)
    cv2.waitKey()
