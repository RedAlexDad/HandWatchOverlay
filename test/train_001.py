import cv2
import numpy as np

def detect_and_save():
    alpha = 0.2
    beta = 1 - alpha
    cap = cv2.VideoCapture(0)
    im_height = 50  # define your top image size here
    im_width = 50
    im = cv2.resize(cv2.imread("watch1.png"), (im_width, im_height))

    classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    while (True):
        ret, frame = cap.read()
        frame[0:im_width,
        0:im_height] = im  # for top-left corner, 0:50 and 0:50 for my image; select your region here like 200:250

        cv2.imshow("live camera", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_and_save()