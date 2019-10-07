import cv2, numpy
import face_detection, face_base_read


if __name__ == '__main__':
    cascade = face_detection.cascade_init("haarcascade_frontalface_default.xml")

    imagePathes, discr = face_base_read.get_lfv_images_base()
    print(cascade)
    for i in range(0,len(imagePathes)):
        frame = cv2.imread(imagePathes[i])
        grayFrame = frame[:, :, :]
        grayFrame = cv2.cvtColor(grayFrame, cv2.COLOR_BGR2GRAY)

        for rect in discr[i]:
            topLeft = (rect[0], rect[1])
            downRight = (rect[0] + rect[2], rect[1] + rect[3])
            cv2.rectangle(frame, topLeft, downRight, (255, 0, 0), 2)

        foundFaces = face_detection.detect_face_on_frame_with_haar(grayFrame, cascade, 1.4, 3, (30, 30), cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in foundFaces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

        cv2.imshow("frame", frame)
        cv2.waitKey(500)
