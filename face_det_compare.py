import cv2, numpy, time
import face_detection, face_base_read

def rect_squre(w, h):
    if w < 0 or h < 0:
        return 0
    else:
        return w*h

def rects_compare(x1,y1,w1,h1,x2,y2,w2,h2):
    leftOut = min(x1,x2)
    rightOut = max(x1+w1,x2+w2)
    topOut = min(y1,y2)
    bottomOut = max(y1+h1,y2+h2)

    leftInside = max(x1,x2)
    rightInside = min(x1+w1,x2+w2)
    topInside = max(y1,y2)
    bottomInside = min(y1+h1,y2+h2)

    #print(f'out rect: {leftOut} {topOut} {rightOut} {bottomOut}')
    #print(f'in rect: {leftInside} {topInside} {rightInside} {bottomInside}')
    #print(f'sq coef {rect_squre(rightInside-leftInside,bottomInside-topInside)/rect_squre(rightOut-leftOut,bottomOut-topOut)}')

    return rect_squre(rightInside-leftInside,bottomInside-topInside)/rect_squre(rightOut-leftOut,bottomOut-topOut)

    #outSqure = rect_squre(rightOut-leftOut,bottomOut-topOut)
    #insideSqure = rect_squre(rightInside-leftInside,bottomInside-topInside)

    #print(f'{outSqure} {insideSqure}')
    #return lqeftOut, topOut, rightOut, bottomOut, leftInside, topInside, rightInside, bottomInside


def countPos(row, value=0.0):
    elems = 0
    for elem in row:
        if elem > value:
            elems += 1
    return elems


def situationCount(tab, found):
    tp = 0
    fp = 0
    doubleTP = 0
    objectsForFind = len(tab)
    foundObjects = 0
    if found:
        objectsForFind = len(tab[0])
        foundObjects = len(tab)
        for i in range(0,objectsForFind):
            cow = [tab[j][i] for j in range(0,foundObjects)]
            if max(cow) > 0.5:
                tp += 1
            if countPos(cow,0.5) > 1:
                doubleTP += countPos(cow, 0.5)-1
            if countPos(cow) - countPos(cow,0.5) > 0:
                fp += countPos(cow) - countPos(cow,0.5)
    tn = objectsForFind - tp
    #print(f'Найдено объектов {foundObjects}, необходимо было найти {objectsForFind}')
    return tp, tn, fp,  foundObjects, objectsForFind, doubleTP


def oneIteration(cascade, scale = 1.01):
    imagePathes, discr = face_base_read.get_lfv_images_base()
    tpCount = 0
    tnCount = 0
    fpCount = 0
    forFindCount = 0
    foundCount = 0
    doubleTPCount = 0

    allTime = time.time()

    for i in range(0, len(imagePathes)):
        frame = cv2.imread(imagePathes[i])
        grayFrame = frame[:, :, :]
        grayFrame = cv2.cvtColor(grayFrame, cv2.COLOR_BGR2GRAY)

        for rect in discr[i]:
            topLeft = (rect[0], rect[1])
            downRight = (rect[0] + rect[2], rect[1] + rect[3])
            cv2.rectangle(frame, topLeft, downRight, (255, 0, 0), 2)

        foundFaces = face_detection.detect_face_on_frame_with_haar(grayFrame, cascade, scale, 3, (30, 30), cv2.CASCADE_SCALE_IMAGE)

        tabX = [0] * len(discr[i])
        tab = [0] * len(foundFaces)
        for k in range(0, len(foundFaces)):
            tab[k] = tabX[:]

        m = 0
        for (x, y, w, h) in foundFaces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
            n = 0
            for rect in discr[i]:
                tab[m][n] = rects_compare(x, y, w, h, rect[0], rect[1], rect[2], rect[3])
                # print(f'Параметры двух изображений с номером {n} {m}: {x,y,w,h,rect[0],rect[1],rect[2],rect[3]}. Результат равен {tab[m][n]}')
                n += 1
            # print(f'весь ряд: {tab[m]}')
            m += 1

        found = True

        if not tab:
            tab = [0] * len(tabX)
            found = False

        #print(f'{i} из {len(imagePathes)} результат {tab}')

        tpSt, tnSt, fpSt, foundSt, forFindSt, doubleTPSt = situationCount(tab, found)
        tpCount += tpSt
        tnCount += tnSt
        fpCount += fpSt
        foundCount += foundSt
        forFindCount += forFindSt
        doubleTPCount += doubleTPSt
        # print(f'{tpSt} {foundSt} {forFindSt} {doubleTPSt}')
        # print(f'{tpCount} {foundCount} {forFindCount} {doubleTPCount}')

        # cv2.imshow("frame", frame)
        # cv2.waitKey(1)

    timeFin = round((time.time() - allTime) / len(imagePathes) * 1000, 4)
    #print(f'Верно найдено {tpCount}, неверно найденные {fpCount}, нейнайденные {tnCount},  требовалось найти {forFindCount}, найдено {foundCount}, двойных срабатываний {doubleTPCount}')
    #print(f'Время: {timeFin} ms per frame, tp {tpCount} fp {fpCount} tn {tnCount} double {doubleTPCount}')
    return timeFin, tpCount, fpCount, tnCount, doubleTPCount

if __name__ == '__main__':
    cascade = face_detection.cascade_init("haarcascade_frontalface_default.xml")
    scale = 1.05
    while scale < 2:
        timeFin, tpCount, fpCount, tnCount, doubleTPCount = oneIteration(cascade, scale)
        print(f'Параметры: Scale {scale} - Результаты: время {timeFin} ms per frame, tp {tpCount} fp {fpCount} tn {tnCount} double {doubleTPCount}')
        scale += 0.05