import cv2, os

def lfw_discription(path):
    pathDiscription = f'{path}/FDDB-folds'
    return pathDiscription

#0 - rectangles
#1 - ellipses
def get_lfv_images_base(type = 0, debug = 0):
    os.chdir(lfw_discription("../"))
    fileDiscrPath = os.getcwd()

    # this code create two lists of files: with images names and with images discriptions
    filesList = os.listdir(os.getcwd())
    filesName = [filesList[i] for i in range(0, len(filesList)) if i % 2 != 0]
    filesDiscrImage = [filesList[i] for i in range(0, len(filesList)) if i % 2 == 0]

    fullNames = []
    fullDiscriptions = []

    # checking each files
    for i in range(0, len(filesName)):
        nameFiles = fileDiscrPath + "/" + filesName[i]

        imagesPathes = []
        imagesNames = []
        filesDiscriptions = []
        imagesDiscriptions = []

        with open(nameFiles, 'r') as f:
            for line in f:
                imagesPathes.append("../originalPics/" + line[:-1] + ".jpg")
                imagesNames.append(line[:-1])
        nameDiscriptors = fileDiscrPath + "/" + filesDiscrImage[i]
        with open(nameDiscriptors, 'r') as f:
            for line in f:
                filesDiscriptions.append(line[:-1])

        for j in range(0, len(imagesPathes)):
            k = filesDiscriptions.index(imagesNames[j])
            featuresList = []
            for l in range(k + 2, k + 2 + int(filesDiscriptions[k + 1])):
                tempList = filesDiscriptions[l].split(" ")
                if type == 1:
                    featuresList.append([int(float(tempList[3])), int(float(tempList[4])), int(float(tempList[0])), int(float(tempList[1])), int(float(tempList[2]))])
                elif type == 0:
                    topLeft_x = int(float(tempList[3])) - int(float(tempList[1]))
                    topLeft_y = int(float(tempList[4])) - int(float(tempList[0]))
                    tempList = [topLeft_x, topLeft_y, 2 * int(float(tempList[1])), 2 * int(float(tempList[0]))]
                    featuresList.append(tempList)
            imagesDiscriptions.append(featuresList)

        if debug:
            frame = cv2.imread(imagesPathes[0])
            for j in range(0, len(imagesDiscriptions[0])):
                workDiscriptions = imagesDiscriptions[0][j]
                if type == 0:
                    topLeft = (workDiscriptions[0], workDiscriptions[1])
                    downRight = (workDiscriptions[0] + workDiscriptions[2], workDiscriptions[1] + workDiscriptions[3])
                    cv2.rectangle(frame, topLeft, downRight, (255, 0, 0), 2)
                elif type == 1:
                    cv2.ellipse(frame, (workDiscriptions[0], workDiscriptions[1]), (workDiscriptions[3], workDiscriptions[2]), workDiscriptions[4], 0, 360, (0, 255, 0), 1)
            cv2.imshow("name", frame)
            cv2.waitKey(900)

        fullNames += imagesPathes
        fullDiscriptions += imagesDiscriptions

    if debug:
        cv2.destroyAllWindows()
    return fullNames, fullDiscriptions

if __name__ == '__main__':
    path, discr = get_lfv_images_base(0,1)
    print(path)
    print(discr)


