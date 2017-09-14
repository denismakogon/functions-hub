import cv2

facePath = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
smilePath = "/usr/local/share/OpenCV/haarcascades/haarcascade_smile.xml"
faceCascade = cv2.CascadeClassifier(facePath)
smileCascade = cv2.CascadeClassifier(smilePath)


def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.01,
        minNeighbors=8,
        minSize=(55, 55),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces):
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            smile = smileCascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.7,
                minNeighbors=22,
                minSize=(25, 25),
                flags=cv2.CASCADE_SCALE_IMAGE,
                )

            if len(smile):
                for (x_smile, y_smile, w_smile, h_smile) in smile:
                    cv2.rectangle(roi_color,
                                  (x_smile, y_smile),
                                  (x_smile + w_smile, y_smile + h_smile),
                                  (255, 0, 0), 1)

            return len(smile) != 0, frame
