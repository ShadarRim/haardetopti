import cv2, os

def lfw_discription(path):
    pathDiscription = f'{path}/FDDB-folds'
    return pathDiscription

if __name__ == '__main__':
    os.chdir(lfw_discription("../"))
    fileDiscrPath = os.getcwd()
    filesList = os.listdir(os.getcwd())
    filesName = [filesList[i] for i in range(0,len(filesList)) if i%2!=0]
    filesDiscrImage = [filesList[i] for i in range(0,len(filesList)) if i%2==0]

    for i in range(0,len(filesName)):
        nameFiles = fileDiscrPath + "/" + filesName[i]
        imagesPathes = []
        imagesNames = []
        filesDiscriptions = []
        imagesDiscriptions = []
        ellipsisDiscriptions = []
        with open(nameFiles, 'r') as f:
            for line in f:
                imagesPathes.append("../originalPics/"+line[:-1]+".jpg")
                imagesNames.append(line[:-1])
        nameDiscriptors = fileDiscrPath + "/" + filesDiscrImage[i]
        with open(nameDiscriptors, 'r') as f:
            for line in f:
                filesDiscriptions.append(line[:-1])

        for j in range(0,len(imagesPathes)):
            k = filesDiscriptions.index(imagesNames[j])
            tempList = []
            rectangleList = []
            ellipsisList = []
            for l in range(k+2, k + 2 + int(filesDiscriptions[k + 1])):
                tempList = filesDiscriptions[l].split(" ")
                #print(f'{int(float(tempList[3]))} {tempList[4]}')
                topLeft_x = int(float(tempList[3]))-int(float(tempList[0]))
                topLeft_y = int(float(tempList[4]))-int(float(tempList[1]))
                ellipsisList.append([int(float(tempList[3])),int(float(tempList[4])), int(float(tempList[0])),int(float(tempList[1])), int(float(tempList[2]))])

                tempList = [topLeft_x, topLeft_y, 2*int(float(tempList[0])), 2*int(float(tempList[1]))]
                rectangleList.append(tempList)
            imagesDiscriptions.append(rectangleList)
            ellipsisDiscriptions.append(ellipsisList)

        print(len(imagesPathes))
        #print(len(filesDiscriptions))
        print(len(imagesDiscriptions))
        frame = cv2.imread(imagesPathes[0])
        for j in range(0,len(imagesDiscriptions[0])):
            print(imagesDiscriptions[0])
            workDiscriptions = imagesDiscriptions[0][j]
            elWorkDiscr = ellipsisDiscriptions[0][j]
            topLeft = (workDiscriptions[0], workDiscriptions[1])
            downRight = (workDiscriptions[0] + workDiscriptions[2], workDiscriptions[1] + workDiscriptions[3])
            cv2.rectangle(frame, topLeft, downRight, (255, 0, 0), 2)
            cv2.ellipse(frame, (elWorkDiscr[0],elWorkDiscr[1]), (elWorkDiscr[2], elWorkDiscr[3]), elWorkDiscr[4], 0, 180, (0, 255, 0), 1)

        cv2.imshow("name", frame)
        cv2.waitKey(900)

    cv2.destroyAllWindows()



