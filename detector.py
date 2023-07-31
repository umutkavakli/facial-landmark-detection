import os
import cv2
import time
import argparse

CASCADE_PATH = 'haarcascades/'

detectorPaths = {
    "face": "haarcascade_frontalface_default.xml",
    "eyes": "haarcascade_eye.xml",
    "smile": "haarcascade_smile.xml"
}

print("[INFO] Loading haar cascades...")

parser = argparse.ArgumentParser(
    prog="Face Detector"
)

# specify and get arguments from the user
parser.add_argument('-p', '--path')
parser.add_argument('-e', '--eyes', action='store_true')
parser.add_argument('-s', '--smiles', action="store_true")
args = vars(parser.parse_args())

# if path is specified use the video path, otherwise open webcam
record_type = args["path"] if args["path"] is not None else 0  

# store all detectors using cascades
detectors = {}

for (name, path) in detectorPaths.items():
    path = os.path.join(CASCADE_PATH, path)

    detectors[name] = cv2.CascadeClassifier(path)

cap = cv2.VideoCapture(record_type)
fps = 0

while True:

    start = time.time()
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faceRects = detectors["face"].detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=15, 
        minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faceRects) > 0:
        for (faceX, faceY, faceWidth, faceHeight) in faceRects:
            # extract the face ROI
            faceROI = gray[faceY: faceY + faceHeight, faceX: faceX + faceWidth]
            cv2.rectangle(frame, (faceX, faceY), (faceX + faceWidth, faceY + faceHeight), (0, 255, 0), 2)

            eyesRects = []
            smileRects = []

            if args["eyes"]:
                # apply eyes detection to the face ROI
                eyesRects = detectors["eyes"].detectMultiScale(
                    faceROI, scaleFactor=1.2, minNeighbors=5, 
                    minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE
                )

            if args["smiles"]:
                # apply smile detection to the face ROI
                smileRects = detectors["smile"].detectMultiScale(
                    faceROI, scaleFactor=1.2, minNeighbors=20,
                    minSize=(20, 15), flags=cv2.CASCADE_SCALE_IMAGE
                )

            for (eyeX, eyeY, eyeWidth, eyeHeight) in eyesRects:
                pointA = (faceX + eyeX, faceY + eyeY)
                pointB = (faceX + eyeX + eyeWidth, faceY + eyeY + eyeHeight)

                cv2.rectangle(frame, pointA, pointB, (0, 0, 255), 2)

            for (smileX, smileY, smileWidth, smileHeight) in smileRects:
                pointA = (faceX + smileX, faceY + smileY)
                pointB = (faceX + smileX + smileWidth, faceY + smileY + smileHeight)

                cv2.rectangle(frame, pointA, pointB, (255, 0, 0), 2)
    else:
        cv2.putText(frame, "NO FACES", (240, 240), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, f'FPS: {str(int(fps))}', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Face Detector', frame)

    end = time.time()
    fps = 1 / (end - start)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




