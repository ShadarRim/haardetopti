import cv2, time
import numpy as np

def default_webcamera(resWidth, resHeight):
    webcam = cv2.VideoCapture(0)
    webcam.set(3,resWidth)
    webcam.set(4,resHeight)
    return webcam


if __name__ == '__main__':
    webcam = default_webcamera(1280, 960)
    listOfDurations = []

    while True:
        timeStart = time.time()
        ret, frame = webcam.read()
        if not ret:
            break
        cv2.imshow('Webcam test frame', frame)

        workTimePerFrame = time.time() - timeStart
        if len(listOfDurations) > 100:
            listOfDurations.remove(listOfDurations[0])
        listOfDurations.append(workTimePerFrame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f'Average time per frame : {round(np.mean(listOfDurations)*1000,2)} ms')
    webcam.release()
    cv2.destroyAllWindows()
