import cv2
import numpy
import numpy as np
import time
start_time = time.time()

def empty(a):
    pass

def preprocessImage(trackbarName, threshold1Name, threshold2Name, imgResize, kernelBigSize, kernelSmallSize):
    imgBlur = cv2.GaussianBlur(imgResize, (7,7), 1)
    imgGrey = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    threshold1 = cv2.getTrackbarPos(threshold1Name, trackbarName)
    threshold2 = cv2.getTrackbarPos(threshold2Name, trackbarName)
    imgCanny = cv2.Canny(imgGrey, threshold1, threshold2)
    ## Gdyby nie łapało osi to można zwiększyć kernelBig na 7x7 ##
    kernelBig = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelBigSize,kernelBigSize))
    kernelSmall = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelSmallSize,kernelSmallSize))
    imgClose = cv2.morphologyEx(imgCanny, cv2.MORPH_CLOSE, kernelBig, iterations=1)
    imgErode = cv2.erode(imgClose, kernelSmall)

    return imgBlur, imgGrey, imgCanny, imgClose, imgErode

def drawLines(coef, img, thickness, color):
    a = coef[0]
    b = coef[1]
    rho = coef[2]
    x0 = a * rho
    y0 = b * rho
    x1 = int (x0 + 1000 * (-b))
    y1 = int (y0 + 1000 * a)
    x2 = int (x0 - 1000 * (-b))
    y2 = int (y0 - 1000 * a)
    cv2.line(img, (x1,y1), (x2,y2), color, thickness)

def detectAllLines(thresholdHoughName, imgCanny, trackbarName):
    thresholdHough = cv2.getTrackbarPos(thresholdHoughName, trackbarName)
    lines = cv2.HoughLines(imgCanny, 1, np.pi / 180, thresholdHough)

    coefs = []
    cleanCoefs = []
    avgCoefs = []

    for line in lines:
        rho, theta = line[0]
        a = np.round(np.cos(theta), 5)
        b = np.round(np.sin(theta), 5)

        coefs.append((a,b,rho, theta))

        if len(cleanCoefs) == 0:
            cleanCoefs.append((a,b,rho, theta))

        ## Jeśli znajdziemy nową prostą, dodajemy jej współczynniki do cleanCoefs ##
        newCoef = True
        for coef in cleanCoefs:
            aDiff = np.abs(np.abs((np.pi / 2 - theta)) - np.abs((np.pi / 2 - coef[3])))
            bDiff = np.abs(np.abs(rho) - np.abs(coef[2]))
            if aDiff < 0.07 and bDiff < 20:
                newCoef = False
                break

        if newCoef == True:
            cleanCoefs.append((a,b,rho, theta))

    return coefs, cleanCoefs, avgCoefs

def averageCleanLines(coefs, cleanCoefs, avgCoefs):
    for coef in cleanCoefs:
        coefSum = [0, 0, 0, 0]
        coefQ = 0
        for dirtyCoef in coefs:
            aDiff = np.abs(dirtyCoef[0] - coef[0])
            bDiff = np.abs(dirtyCoef[1] - coef[1])
            if aDiff < 0.08 and bDiff < 0.08:
                coefSum [0] += dirtyCoef[0]
                coefSum [1] += dirtyCoef[1]
                coefSum [2] += dirtyCoef[2]
                coefSum [3] += dirtyCoef[3]
                coefQ += 1
        avgCoefs.append((np.round(coefSum[0]/coefQ, 3), np.round(coefSum[1]/coefQ, 3), np.round(coefSum[2]/coefQ, 3), np.round(coefSum[3]/coefQ, 3)))
    return avgCoefs

def getPerpendicularLines(avgCoefs):
    perpCoefs = avgCoefs.copy()
    toRemove = []
    for coef in perpCoefs:
        perpFriend = False
        for i in range(len(perpCoefs)):
            ## Jeśli wśród coefów znajduje się jakiś o theta odchylonym o około pi/2 to przechodzimy dalej ##
            if np.abs(np.abs(coef[3] - perpCoefs[i][3]) - np.pi / 2) < 0.12 and perpCoefs.index(coef) != i:
                perpFriend = True

        ## Jeśli nie, dodajemy współczynnik do usunięcia ##
        if not perpFriend:
            toRemove.append(coef)

    for coef in toRemove:
        perpCoefs.remove(coef)

    if len(perpCoefs) == 1:
        perpCoefs = []

    return perpCoefs

def getXandYaxis(perpCoefs):
    xAxis = []
    yAxis = []
    if len(perpCoefs) == 2:
        if np.abs(perpCoefs[0][3] - np.pi / 2) < np.abs(perpCoefs[1][3] - np.pi / 2):
        # if perpCoefs[0][3] < perpCoefs[1][3]:
            xAxis = perpCoefs[0]
            yAxis = perpCoefs[1]
        else:
            xAxis = perpCoefs[1]
            yAxis = perpCoefs[0]
    return xAxis, yAxis

def detectAllParallelLines(thresholdHoughName, imgCanny, trackbarName):
    thresholdHough = cv2.getTrackbarPos(thresholdHoughName, trackbarName)
    lines = cv2.HoughLines(imgCanny, 1, np.pi / 180, thresholdHough)

    coefs = []
    parallelCoefs = []
    cleanParallelCoefs = []
    avgCoefs = []

    for line in lines:
        rho, theta = line[0]
        a = np.round(np.cos(theta), 5)
        b = np.round(np.sin(theta), 5)

        coefs.append((a,b,rho, theta))

        if len(parallelCoefs) == 0 and np.abs(a) < 0.5:
            parallelCoefs.append((a,b,rho, theta))

        ## Jeśli znajdziemy nową prostą równoległą, dodajemy jej współczynniki do parallelCoefs ##
        newCoef = True
        newCleanCoef = True

        for coef in parallelCoefs:
            aDiff = np.abs(theta - coef[3])
            bDiff = np.abs(rho - coef[2])
            # print("bDiff: " + str(bDiff))
            if aDiff > 0.1:
                newCoef = False
                break
            if bDiff < 10:
                newCleanCoef = False

        if newCoef == True:
            parallelCoefs.append((a,b,rho, theta))
            if newCleanCoef == True:
                cleanParallelCoefs.append((a,b,rho, theta))

    ## Jeśli wykryto tylko jedną linię, to prawdopodobnie jest to funkcja, nierównoległa do żadnych kratek ##
    ## Powtarzamy procedurę pomijając pierwszy element ##
    if len(cleanParallelCoefs) == 0:
        print("Nie wykryto żadnej linii przy analizie skali")
    else:
        if len(cleanParallelCoefs) == 1:
            thresholdHough = cv2.getTrackbarPos(thresholdHoughName, trackbarName)
            lines = cv2.HoughLines(imgCanny, 1, np.pi / 180, thresholdHough)
            numpy.delete(lines, 0, axis=None)

            coefs = []
            parallelCoefs = []
            cleanParallelCoefs = []
            avgCoefs = []

            for line in lines:
                rho, theta = line[0]
                a = np.round(np.cos(theta), 5)
                b = np.round(np.sin(theta), 5)

                coefs.append((a,b,rho, theta))

                if len(parallelCoefs) == 0 and np.abs(a) < 0.5:
                    parallelCoefs.append((a,b,rho, theta))

                ## Jeśli znajdziemy nową prostą równoległą, dodajemy jej współczynniki do parallelCoefs ##
                newCoef = True
                newCleanCoef = True

                for coef in parallelCoefs:
                    aDiff = np.abs(a - coef[0])
                    bDiff = np.abs(rho - coef[2])
                    # print("bDiff: " + str(bDiff))
                    if aDiff > 0.02:
                        newCoef = False
                        break
                    if bDiff < 10:
                        newCleanCoef = False

                if newCoef == True:
                    parallelCoefs.append((a,b,rho, theta))
                    if newCleanCoef == True:
                        cleanParallelCoefs.append((a,b,rho, theta))

    return coefs, parallelCoefs, cleanParallelCoefs, avgCoefs

def averageCleanParallelLines(coefs, parallelCoefs, cleanParallelCoefs, avgParallelCoefs):
    for coef in cleanParallelCoefs:
        coefSum = [0, 0, 0, 0]
        coefQ = 0
        for dirtyCoef in parallelCoefs:
            aDiff = np.abs(dirtyCoef[0] - coef[0])
            bDiff = np.abs(dirtyCoef[2] - coef[2])
            if aDiff < 0.02 and bDiff < 10:
                coefSum [0] += dirtyCoef[0]
                coefSum [1] += dirtyCoef[1]
                coefSum [2] += dirtyCoef[2]
                coefSum [3] += dirtyCoef[3]
                coefQ += 1
        avgParallelCoefs.append((np.round(coefSum[0]/coefQ, 3), np.round(coefSum[1]/coefQ, 3), np.round(coefSum[2]/coefQ, 3), np.round(coefSum[3]/coefQ, 3)))
    return avgParallelCoefs

def getDstBtwnLines(avgParallelCoefs):
    ## Sortowanie linii rosnąco ##
    def takeThird(elem):
        return elem[2]

    avgParallelCoefs.sort(key=takeThird)

    print(avgParallelCoefs)

    ## Liczenie średniej odległości między wykrytymi liniami ##
    dstSum = 0
    for i in range(len(avgParallelCoefs) - 1):
        dstSum += avgParallelCoefs[i+1][2] - avgParallelCoefs[i][2]

    avDst = dstSum / (len(avgParallelCoefs) - 1)

    ## Liczenie średniej odległości między liniami, z pominięciem tych dalszych niż 1.2 * avDst ##
    dstCleanSum = 0
    quantity = 0
    for i in range(len(avgParallelCoefs) - 1):
        if avgParallelCoefs[i+1][2] - avgParallelCoefs[i][2] < 1.2 * avDst:
            quantity +=1
            dstCleanSum += avgParallelCoefs[i+1][2] - avgParallelCoefs[i][2]

    avCleanDst = dstCleanSum / quantity

    return avCleanDst

def getCoefsLinearFunction(avgCoefs, imgResize, perpCoefs, xAxis, yAxis):
    functionParams = avgCoefs.copy()
    functionParams.remove(perpCoefs [0])
    functionParams.remove(perpCoefs [1])
    functionParams = functionParams[0]

    # Rysuj funkcję liniową ##
    imgLinearFunction = imgResize.copy()
    coef = functionParams
    drawLines(coef, imgLinearFunction, 1, 255)

    ## Liczenie wspóczynnika a ##
    thetaDiff = np.round(functionParams[3] - yAxis[3], 5)
    relativeA = np.round((1 / np.tan(thetaDiff)), 2)

    ## Liczenie współczynnika b ##
    rF = functionParams[2]
    tF = functionParams[3]
    rX = xAxis[2]
    tX = xAxis[3]
    rY = yAxis[2]
    tY = yAxis[3]

    ## Gdyby któraś z osi była idealnie równoległa do poziomu/pionu unikamy dzielenia przez 0 ##
    if tY == 0:
        tY = 0.001
    if tX == 0:
        tX = 0.001

    x1 = ((rY / np.sin(tY)) - (rF / np.sin(tF))) / ((1 / np.tan(tY)) - (1 / np.tan(tF)))
    y1 = - x1 / np.tan(tF) + rF / np.sin(tF)

    x0 = ((rY / np.sin(tY)) - (rX / np.sin(tX))) / ((1 / np.tan(tY)) - (1 / np.tan(tX)))
    y0 = - x0 / np.tan(tX) + rX / np.sin(tX)

    bAbs = np.sqrt((x1-x0)**2 + (y1-y0)**2)

    if y1 < y0:
        b = bAbs
    else:
        b = - bAbs

    bScaled = round(b / scale, 2)

    if bScaled >= 0:
        print("****************************************************************")
        print("Rozpoznano funkcję liniową o wzorze: y = " + str(relativeA) + "x + " + str(bScaled))
        print("****************************************************************")
    else:
        print("****************************************************************")
        print("Rozpoznano funkcję liniową o wzorze: y = " + str(relativeA) + "x " + str(bScaled))
        print("****************************************************************")

    return imgLinearFunction, relativeA, bScaled

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def createLabel(steps, min, max):
    label = np.zeros(steps+1)
    dist = (max - min) / steps
    for i in range(len(label)):
        label[i] = min + (i * dist)
    length = steps + 1
    return label, dist, length

# def createRanges(steps, min, max):
#     ranges = np.zeros(steps+2)
#     dist = (max - min) / steps
#     for i in range(len(ranges)):
#         ranges[i] = min + (i * dist) - (dist / 2)
#     return ranges

def getCenterPoint(xAxis, yAxis):
    rX = xAxis[2]
    tX = xAxis[3]
    rY = yAxis[2]
    tY = yAxis[3]

    ## Gdyby któraś z osi była idealnie równoległa do poziomu/pionu unikamy dzielenia przez 0 ##
    if tY == 0:
        tY = 0.001
    if tX == 0:
        tX = 0.001

    ## Wziąć pod uwagę obrót ##
    y0 = ((rY / np.sin(tY)) - (rX / np.sin(tX))) / ((1 / np.tan(tY)) - (1 / np.tan(tX)))
    x0 = - y0 / np.tan(tX) + rX / np.sin(tX)

    return x0, y0

def getCoefsAndAxes(imgResize, thresholdHoughName, trackbarName, imgErode):
    ## HOUGH - wykrywamy oś OX oraz OY ##
    ## Wykrywanie wszystkich linii i nakładanie ich na obraz ##
    coefs, cleanCoefs, avgCoefs  = detectAllLines(thresholdHoughName, imgErode, trackbarName)
    imgAllLines = imgResize.copy()
    for coef in coefs:
        drawLines(coef, imgAllLines, 1, 255)


    ## Uśrednianie duplikujących się linii i nakładanie ich na obraz ##
    avgCoefs = averageCleanLines(coefs, cleanCoefs, avgCoefs)
    imgAvgLines = imgResize.copy()
    for coef in avgCoefs:
        drawLines(coef, imgAvgLines, 1, 255)


    ## Wyciąganie prostopadłych linii ##
    perpCoefs = getPerpendicularLines(avgCoefs)
    imgPerpLines = imgResize.copy()
    for coef in perpCoefs:
        drawLines(coef, imgPerpLines, 1, 255)

    ## Parametryzacja osi OX oraz OY ##
    xAxis, yAxis = getXandYaxis(perpCoefs)
    if len(xAxis)!= 0:
        imgXAxis = imgResize.copy()
        drawLines(xAxis, imgXAxis, 1, 255)
        imgYAxis = imgResize.copy()
        drawLines(yAxis, imgYAxis, 1, 255)
    else:
        imgXAxis = imgResize.copy()
        imgYAxis = imgResize.copy()
        print("Wystąpił problem ze znalezieniem osi.")

    return imgAllLines, imgAvgLines, imgPerpLines, imgXAxis, imgYAxis, coefs, cleanCoefs, avgCoefs, perpCoefs, xAxis, yAxis

def getOnlyParabole(imgParaboleStraight, xAxis, yAxis, width, height, kernelBigSize, kernelSmallSize):
    ## Usuwanie osi współrzędnych ##
    drawLines(xAxis, imgParaboleStraight, 5, 0)
    drawLines(yAxis, imgParaboleStraight, 5, 0)

    ## Wykrywanie wszystkich konturów ##
    conts, hier = cv2.findContours(imgParaboleStraight, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    imgParaboleConts = imgResize.copy()
    imgParaboleConts = cv2.drawContours(imgParaboleConts, conts, -1, (255, 0, 255), 1)

    ## Wykrywanie dobrych konturów ##
    goodConts = []
    for cont in conts:
      area = cv2.contourArea(cont)
      if area > 50:
        goodConts.append(cont)

    imgParaboleContsGood = np.zeros((width, height), np.uint8)
    imgParaboleContsGood = cv2.drawContours(imgParaboleContsGood, goodConts, -1, 255, 1)

    ## Wypełnianie dobrych konturów ##
    kernelBig = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelBigSize,kernelBigSize))
    kernelSmall = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelSmallSize,kernelSmallSize))
    imgParaboleContsClose = cv2.morphologyEx(imgParaboleContsGood, cv2.MORPH_CLOSE, kernelBig, iterations=1)
    imgParaboleContsErode = cv2.erode(imgParaboleContsClose, kernelSmall)

    return conts, hier, imgParaboleConts, goodConts, imgParaboleStraight, imgParaboleContsGood, imgParaboleContsClose, imgParaboleContsErode

def rotateParabole(imgErode, yAxis, imgResize):
    imgParabole = imgErode.copy()

    ## Ustawiamy parabolę tak, jak osie obrazu ##
    if yAxis[3] > np.pi / 2:
        imgParaboleStraight = rotateImage(imgParabole, (yAxis[3] - np.pi) * 180 / np.pi)
        imgParaboleStraight = rotateImage(imgParaboleStraight, 270)
        imgResize = rotateImage(imgResize, (yAxis[3] - np.pi) * 180 / np.pi)
        imgResize = rotateImage(imgResize, 270)
    else:
        imgParaboleStraight = rotateImage(imgParabole, yAxis[3] * 180 / np.pi)
        imgParaboleStraight = rotateImage(imgParaboleStraight, 270)
        imgResize = rotateImage(imgResize, yAxis[3] * 180 / np.pi)
        imgResize = rotateImage(imgResize, 270)

    return imgParaboleStraight, imgResize

def houghParabole(imgParaboleStraight, aMin, aMax, bMin, bMax, cMin, cMax, aSteps, bSteps, cSteps, xAxis, yAxis):
    rows = len(imgParaboleStraight)
    cols = len(imgParaboleStraight[0])

    votes = np.zeros((aSteps+1, bSteps+1, cSteps+1))
    aLabel, aDist, aLen = createLabel(aSteps, aMin, aMax)
    bLabel, bDist, bLen = createLabel(bSteps, bMin, bMax)
    cLabel, cDist, cLen = createLabel(cSteps, cMin, cMax)

    x0, y0 = getCenterPoint(xAxis, yAxis)
    print("Center point: " + str(x0) + " " + str(y0))

    for x in range(cols):
        xRel = (x - x0) / scale
        xx = xRel*xRel
        for y in range(rows):
            yRel = (y - y0) / scale
            aValue = aMin

            if imgParaboleStraight[x][y] == 255:
                for a in range(aLen):
                    axx = aValue*xx
                    bValue = bMin

                    for b in range(bLen):
                        bx = bValue*xRel
                        cValue = cMin

                        for c in range(cLen):
                            ## Liczenie oczekiwanego y tak, aby punkt należał do danej paraboli ##
                            yExpected = axx + bx + cValue

                            if yRel - yExpected < 0.02 and yRel - yExpected > - 0.02:
                                votes[a][b][c] += 1

                            cValue += cDist
                        bValue += bDist
                    aValue += aDist

    return x0, y0, aLabel, bLabel, cLabel, votes

def findMaxVotes(votes):
    max = 0
    imax, jmax, kmax = 0,0,0
    for i in range(len(votes[0][0])):
        for j in range(len(votes[0])):
            for k in range(len(votes)):
                if votes[i][j][k] > max:
                    max = votes[i][j][k]
                    imax = i
                    jmax = j
                    kmax = k
    print("****************************************************************")
    print("Znaleziono funkcję kwadratową o wzorze: " + str(round(aLabel[imax], 2)) + " x^2 + " + str(round(bLabel[jmax], 2)) + " x + " + str(round(cLabel[kmax], 2)))
    print("****************************************************************")
    return

## Zdjęcia do dyspozycji ##
imgLiniowa = cv2.imread("liniowa.png")
imgKwadratowa = cv2.imread("kwadratowa.jpg")
imgLiniowaKrzywo = cv2.imread("liniowaKrzywo.png")
imgKwadratowaKrzywo = cv2.imread("KwadratowaKrzywo.jpg")
imgLiniowaNowa = cv2.imread("liniowaNowa.jpg")
imgLiniowaNowaKrzywo = cv2.imread("liniowaNowaKrzywo.jpg")
imgKwadratowaNowa = cv2.imread("kwadratowaNowa.jpg")
imgKwadratowaNowaKrzywo = cv2.imread("kwadratowaNowaKrzywo.jpg")

img = imgLiniowaKrzywo

width, height = 300, 300

imgResize = cv2.resize(img, (width, height))

## Trackbar do dostosowywania operacji na wykresie ##
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 143, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 87, 255, empty)
cv2.createTrackbar("ThresholdHough", "Parameters", 140, 255, empty)
cv2.createTrackbar("ThresholdHoughParabole", "Parameters", 80, 255, empty)

## Trackbar do dostosowywania operacji na kratkach ##
cv2.namedWindow("Kratki")
cv2.resizeWindow("Kratki", 640, 240)
cv2.createTrackbar("Threshold1Checkered", "Kratki", 41, 255, empty)
cv2.createTrackbar("Threshold2Checkered", "Kratki", 16, 255, empty)
cv2.createTrackbar("ThresholdHoughCheckered", "Kratki", 157, 255, empty)

    ## Wstępna obróbka + wyciąganie konturów ##
imgBlur, imgGrey, imgCanny, imgClose, imgErode = preprocessImage("Parameters", "Threshold1", "Threshold2", imgResize, 6, 3)
imgsPreprocess = np.hstack((imgGrey, imgCanny, imgClose, imgErode))
cv2.imshow("Image preprocessing", imgsPreprocess)

## Rozpoznawanie linii oraz parametryzacja osi ##
imgAllLines, imgAvgLines, imgPerpLines, imgXAxis, imgYAxis, coefs, cleanCoefs, avgCoefs, perpCoefs, xAxis, yAxis = getCoefsAndAxes(imgResize, "ThresholdHough", "Parameters", imgErode)
imgsLines = np.hstack((imgAllLines, imgAvgLines, imgPerpLines, imgXAxis, imgYAxis))
cv2.imshow("Line and axes detecion", imgsLines)

#################### SZUKANIE SKALI OBRAZU ##########################
## Wyciąganie kratek ##
imgBlurCheckered, imgGreyCheckered, imgCannyCheckered, imgCloseCheckered, imgErodeCheckered = preprocessImage("Kratki", "Threshold1Checkered", "Threshold2Checkered", imgResize, 3, 2)
imgsGrid = np.hstack((imgGreyCheckered, imgCannyCheckered, imgCloseCheckered, imgErodeCheckered))
cv2.imshow("Grid detection", imgsGrid)

## Wykrywanie równoległych linii ##
coefsCheckered, parallelCoefsCheckered, cleanParallelCoefsCheckered, avgCoefsCheckered = detectAllParallelLines("ThresholdHoughCheckered", imgCannyCheckered, "Kratki")
imgAllLinesCheckered = imgResize.copy()
for coef in parallelCoefsCheckered:
    drawLines(coef, imgAllLinesCheckered, 1, 255)
imgCleanLinesCheckered = imgResize.copy()
for coef in cleanParallelCoefsCheckered:
    drawLines(coef, imgCleanLinesCheckered, 1, 255)

## Uśrednianie równoległych linii #
avgParallelCoefsCheckered = averageCleanParallelLines(coefsCheckered, parallelCoefsCheckered, cleanParallelCoefsCheckered, avgCoefsCheckered)
imgAvgParallelLinesCheckered = imgResize.copy()
for coef in avgParallelCoefsCheckered:
    drawLines(coef, imgAvgParallelLinesCheckered, 1, 255)

## Obliczanie skali - odległości w px między liniami ##
scale = getDstBtwnLines(avgParallelCoefsCheckered)
print("Skala obrazu (ilość pixeli między kratkami) to: " + str(scale))

imgsScale = np.hstack((imgAllLinesCheckered, imgCleanLinesCheckered, imgAvgParallelLinesCheckered))
cv2.imshow("Scale detection", imgsScale)

############################## DZIAŁANIE NA PODSTAWIE TYPU WYKRYTEJ FUNKCJI ################################
# Rozpoznanie, z jaką funkcją mamy do czynienia ##
if len(perpCoefs) != 2:
    print("Nie wykryto osi OX, OY lub masz więcej niż dwie prostopadłe linie na wykresie!")
else:
    ## Akcja, jeżeli wykryto funkcję liniową ##
    if len(avgCoefs) == 3:
        imgLinearFunction, relativeA, bScaled = getCoefsLinearFunction(avgCoefs, imgResize, perpCoefs, xAxis, yAxis)

    ## Akcja, jeżeli nie wykryto funkcji liniowej (w domyśle mamy kwadratową) ##
    if len(avgCoefs) == 2:
        ## Obrócenie paraboli do właściwej pozycji ##
        imgParaboleStraight, imgResize = rotateParabole(imgErode, yAxis, imgResize)

        ## Odczytanie nowych parametrów osi OX, OY ##
        imgAllLines, imgAvgLines, imgPerpLines, imgYAxis, imgXAxis, coefs, cleanCoefs, avgCoefs, perpCoefs, yAxis, xAxis = getCoefsAndAxes(imgResize, "ThresholdHough", "Parameters", imgParaboleStraight)
        imgsLines = np.hstack((imgAllLines, imgAvgLines, imgPerpLines, imgXAxis, imgYAxis))
        cv2.imshow("Line and axes detecion after rotation", imgsLines)

        ## Pozbycie się osi i niechcianych śmieci - zostawiamy samą parabolę ##
        conts, hier, imgParaboleConts, goodConts, imgParaboleStraight, imgParaboleContsGood, imgParaboleContsClose, imgParaboleContsErode = getOnlyParabole(imgParaboleStraight, xAxis, yAxis, width, height, 5, 3)
        imgsCleanParabole = np.hstack((imgErode, imgParaboleStraight, imgParaboleContsGood, imgParaboleContsClose, imgParaboleContsErode))
        cv2.imshow("Rotating and cleaning parabole", imgsCleanParabole)

        ## Przeprowadzenie transformaty Hougha dla paraboli ##
        x0, y0, aLabel, bLabel, cLabel, votes = houghParabole(imgParaboleStraight, -2, 2, -5, 5, -2, 2, 50, 100, 50, xAxis, yAxis)

        ## Znalezienie i wypisanie parametrów, które otrzymały najwięcej głosów ##
        findMaxVotes(votes)

print("--- %s seconds ---" % (time.time() - start_time))

cv2.waitKey(0)
