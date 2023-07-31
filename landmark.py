import cv2
import dlib
import time
import utils
import argparse

LANDMARK_PATH = "landmarks"

parser = argparse.ArgumentParser(
    prog="Face Detector"
)

parser.add_argument('-p', '--path')
args = vars(parser.parse_args())

# if path is specified use the video path, otherwise open webcam
record_type = args["path"] if args["path"] is not None else 0  

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(f"{LANDMARK_PATH}/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(record_type)
fps = 0

while True:

    start = time.time()
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    if len(rects) > 0:
        for (i, rect) in enumerate(rects):

            shape = predictor(gray, rect)
            shape = utils.shape_to_numpy(shape)

            (x, y, w, h) = utils.rect_to_bbox(rect)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.putText(frame, f"Face #{i+1}", (x - 10, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0), 2)

            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    else:
        cv2.putText(frame, "NO FACES", (240, 240), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, f'FPS: {str(int(fps))}', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Landmark Detector", frame)
    
    end = time.time()
    fps = 1 / (end - start)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()