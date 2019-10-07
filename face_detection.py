import cv2, time, camera
import numpy as np


def cascade_init(path):
    return cv2.CascadeClassifier(path)


def detect_face_on_frame_with_haar(grayFrame, faceCascade, scaleFactor, minNeighbours, minSize, flags):
    return faceCascade.detectMultiScale(grayFrame, scaleFactor, minNeighbours, flags, minSize)

if __name__ == '__main__':
    camera = camera.default_webcamera(1280, 960)
    cascade = cascade_init("haarcascade_frontalface_default.xml")
    listOfDurations = []

    allTime = time.time()
    i = 0
    while True:
        i += 1
        timeStart = time.time()
        ret, frame = camera.read()
        if not ret:
            break
        grayFrame = frame[:, :, :]
        cv2.cvtColor(grayFrame, cv2.COLOR_BGR2GRAY)

        faces = detect_face_on_frame_with_haar(grayFrame, cascade, 1.4, 3, (30, 30), cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Webcam test frame', frame)

        workTimePerFrame = time.time() - timeStart

        if len(listOfDurations) > 100:
            listOfDurations.remove(listOfDurations[0])
        listOfDurations.append(workTimePerFrame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f'Average time per frame: {round(np.mean(listOfDurations) * 1000, 2)} ms')
    camera.release()
    cv2.destroyAllWindows()