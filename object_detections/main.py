import cv2
import os

try:
    from .licence_plates import detect as licence_detect
    from .face_features import detect as face_detect
except Exception:
    from licence_plates import detect as licence_detect
    from face_features import detect as face_detect


if __name__ == "__main__":

    cwd = os.getcwd()
    plate_path = os.path.join(cwd, "images/licence_plates/1.png")
    plate_text, frame = licence_detect.detect(cv2.imread(plate_path))
    print("Licence plate: ", plate_text)
    # cv2.imshow("licence plate detection", frame)
    # cv2.waitKey()

    smile_path = os.path.join(cwd, "images/smiles/5.jpg")
    has_smile, frame = face_detect.detect_face(cv2.imread(smile_path))
    print("Has smile? ", "Yes" if has_smile else "No")
    # cv2.imshow("smile detection", frame)
    # cv2.waitKey()
