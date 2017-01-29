import numpy as np
from PIL import Image
import time

"""functies bedoeld voor het testen"""
def imageFrom2D(array):
    im = []
    for y in xrange(0, array.shape[1]):
        row = []
        for x in xrange(0, array.shape[0]):
            row.append([array[y][x], array[y][x], array[y][x]])
        im.append(row)
    return Image.fromarray(np.asarray(im, 'uint8'))
def normalize(array):
    array.flags.writeable = True
    high = 0
    low = 1000000
    for y in xrange(0, array.shape[1]):
        if max(array[y]) > high:
            high = max(array[y])
        if min(array[y]) < low:
            low = min(array[y])
    dif = high - low
    if dif < 1:
        dif = 1
    for y in xrange(0, array.shape[1]):
        for x in xrange(0, array.shape[0]):
            array[y][x] = (array[y][x] - low)*(255/dif)
    return array
def generateModel():
    return np.asarray([

       [np.asarray(np.random.rand(25)),
        np.asarray(np.random.rand(25)),
        np.asarray(np.random.rand(25)),
        np.asarray(np.random.rand(25)),
        np.asarray(np.random.rand(25)),
        np.asarray(np.random.rand(25))],


       [np.asarray(np.random.rand(150)),
        np.asarray(np.random.rand(150)),
        np.asarray(np.random.rand(150)),
        np.asarray(np.random.rand(150)),
        np.asarray(np.random.rand(150)),
        np.asarray(np.random.rand(150))]
    ])
def benchmark(runs):
    starttime = time.time()
    for i in xrange(0, runs):
        cnn('dataset/0/img001-001.png', generateModel(), True)
    now = time.time()
    print "----------------------------------------------------------\nhet convnet heeft over "+str(runs)+" runs " + str(now - starttime)+ "s over gedaan\ndit geeft een gemiddelde van " + str((now - starttime)/runs) + "s per run"


def cnn(imagePath, model, training=False):

    if(training):
        starttime = time.time()

    """plaatje openen"""
    img = np.asarray(Image.open(imagePath))


    """3d plaatje omzetten naar een 2d array met alleen de kleurintensiteit (zwart-wit)"""
    input = []
    for y in range(0, 32):
        row = []
        for x in range(0, 32):
            row.append(max(img[y][x]))
        input.append(row)
    inputlayer = np.asarray(input)


    """eerste featuremaps creeeren. activatiefuncties worden ook meteen toegepast"""
    featuremap1 = [ [], [], [], [], [], [] ]
    convmodel1 = model[0]
    for y in range(2, 30):
        rows = [ [], [], [], [], [], [] ]
        for x in range(2, 30):
            receptivefield = np.asarray([
                inputlayer[y-2][x-2], inputlayer[y-2][x-1], inputlayer[y-2][x], inputlayer[y-2][x+1], inputlayer[y-2][x+2],
                inputlayer[y-1][x-2], inputlayer[y-1][x-1], inputlayer[y-1][x], inputlayer[y-1][x+1], inputlayer[y-1][x+2],
                  inputlayer[y][x-2],   inputlayer[y][x-1],   inputlayer[y][x],   inputlayer[y][x+1], inputlayer[y][x+2],
                inputlayer[y+1][x-2], inputlayer[y+1][x-1], inputlayer[y+1][x], inputlayer[y+1][x+1], inputlayer[y+1][x+2],
                inputlayer[y+2][x-2], inputlayer[y+2][x-1], inputlayer[y+2][x], inputlayer[y+2][x+1], inputlayer[y+2][x+2],
            ])
            for i in range(0, 6):
                rows[i].append(max(0, np.sum(np.multiply(receptivefield, convmodel1[i]))))
        for h in range(0, 6):
            featuremap1[h].append(rows[h])
    featuremap1 = np.asarray(featuremap1)


    """maxpooling toepassen op alle 6 featuremaps"""
    maxpooled1 = [ [], [], [], [], [], [] ]
    for z in range(0, 6):
        collumn = []
        for y in range(0, 28, 2):
            row = []
            for x in range(0, 28, 2):
                row.append(max([featuremap1[z][y][x], featuremap1[z][y][x+1], featuremap1[z][y+1][x], featuremap1[z][y+1][x+1]]))
            collumn.append(row)
        maxpooled1[z] = np.asarray(collumn)
    maxpooled1 = np.asarray(maxpooled1)


    """dropout 1 toepassen"""
    if(training):
        maxpooled1.flags.writeable = True
        for z in range(0, 6):
            for y in range(0, 14):
                for x in range(0, 14):
                    if np.random.rand() < 0.05 :
                        maxpooled1[z][y][x] = 0

    """tweede featuremaps creeeren, activatiefuncties worden ook meteen toegepast"""
    featuremap2 = [ [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []  ]
    convmodel2 = model[1]
    for y in range(2, 12):
        rows = [ [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []  ]
        for x in range(2, 12):
            receptivefield = np.asarray([
                maxpooled1[0][y-2][x-2], maxpooled1[0][y-2][x-1], maxpooled1[0][y-2][x], maxpooled1[0][y-2][x+1], maxpooled1[0][y-2][x+2],
                maxpooled1[0][y-1][x-2], maxpooled1[0][y-1][x-1], maxpooled1[0][y-1][x], maxpooled1[0][y-1][x+1], maxpooled1[0][y-1][x+2],
                  maxpooled1[0][y][x-2],   maxpooled1[0][y][x-1],   maxpooled1[0][y][x],   maxpooled1[0][y][x+1],   maxpooled1[0][y][x+2],
                maxpooled1[0][y+1][x-2], maxpooled1[0][y+1][x-1], maxpooled1[0][y+1][x], maxpooled1[0][y+1][x+1], maxpooled1[0][y+1][x+2],
                maxpooled1[0][y+2][x-2], maxpooled1[0][y+2][x-1], maxpooled1[0][y+2][x], maxpooled1[0][y+2][x+1], maxpooled1[0][y+2][x+2],

                maxpooled1[1][y-2][x-2], maxpooled1[1][y-2][x-1], maxpooled1[1][y-2][x], maxpooled1[1][y-2][x+1], maxpooled1[1][y-2][x+2],
                maxpooled1[1][y-1][x-2], maxpooled1[1][y-1][x-1], maxpooled1[1][y-1][x], maxpooled1[1][y-1][x+1], maxpooled1[1][y-1][x+2],
                  maxpooled1[1][y][x-2],   maxpooled1[1][y][x-1],   maxpooled1[1][y][x],   maxpooled1[1][y][x+1],   maxpooled1[1][y][x+2],
                maxpooled1[1][y+1][x-2], maxpooled1[1][y+1][x-1], maxpooled1[1][y+1][x], maxpooled1[1][y+1][x+1], maxpooled1[1][y+1][x+2],
                maxpooled1[1][y+2][x-2], maxpooled1[1][y+2][x-1], maxpooled1[1][y+2][x], maxpooled1[1][y+2][x+1], maxpooled1[1][y+2][x+2],

                maxpooled1[2][y-2][x-2], maxpooled1[2][y-2][x-1], maxpooled1[2][y-2][x], maxpooled1[2][y-2][x+1], maxpooled1[2][y-2][x+2],
                maxpooled1[2][y-1][x-2], maxpooled1[2][y-1][x-1], maxpooled1[2][y-1][x], maxpooled1[2][y-1][x+1], maxpooled1[2][y-1][x+2],
                  maxpooled1[2][y][x-2],   maxpooled1[2][y][x-1],   maxpooled1[2][y][x],   maxpooled1[2][y][x+1],   maxpooled1[2][y][x+2],
                maxpooled1[2][y+1][x-2], maxpooled1[2][y+1][x-1], maxpooled1[2][y+1][x], maxpooled1[2][y+1][x+1], maxpooled1[2][y+1][x+2],
                maxpooled1[2][y+2][x-2], maxpooled1[2][y+2][x-1], maxpooled1[2][y+2][x], maxpooled1[2][y+2][x+1], maxpooled1[2][y+2][x+2],

                maxpooled1[3][y-2][x-2], maxpooled1[3][y-2][x-1], maxpooled1[3][y-2][x], maxpooled1[3][y-2][x+1], maxpooled1[3][y-2][x+2],
                maxpooled1[3][y-1][x-2], maxpooled1[3][y-1][x-1], maxpooled1[3][y-1][x], maxpooled1[3][y-1][x+1], maxpooled1[3][y-1][x+2],
                  maxpooled1[3][y][x-2],   maxpooled1[3][y][x-1],   maxpooled1[3][y][x],   maxpooled1[3][y][x+1],   maxpooled1[3][y][x+2],
                maxpooled1[3][y+1][x-2], maxpooled1[3][y+1][x-1], maxpooled1[3][y+1][x], maxpooled1[3][y+1][x+1], maxpooled1[3][y+1][x+2],
                maxpooled1[3][y+2][x-2], maxpooled1[3][y+2][x-1], maxpooled1[3][y+2][x], maxpooled1[3][y+2][x+1], maxpooled1[3][y+2][x+2],

                maxpooled1[4][y-2][x-2], maxpooled1[4][y-2][x-1], maxpooled1[4][y-2][x], maxpooled1[4][y-2][x+1], maxpooled1[4][y-2][x+2],
                maxpooled1[4][y-1][x-2], maxpooled1[4][y-1][x-1], maxpooled1[4][y-1][x], maxpooled1[4][y-1][x+1], maxpooled1[4][y-1][x+2],
                  maxpooled1[4][y][x-2],   maxpooled1[4][y][x-1],   maxpooled1[4][y][x],   maxpooled1[4][y][x+1],   maxpooled1[4][y][x+2],
                maxpooled1[4][y+1][x-2], maxpooled1[4][y+1][x-1], maxpooled1[4][y+1][x], maxpooled1[4][y+1][x+1], maxpooled1[4][y+1][x+2],
                maxpooled1[4][y+2][x-2], maxpooled1[4][y+2][x-1], maxpooled1[4][y+2][x], maxpooled1[4][y+2][x+1], maxpooled1[4][y+2][x+2],

                maxpooled1[5][y-2][x-2], maxpooled1[5][y-2][x-1], maxpooled1[5][y-2][x], maxpooled1[5][y-2][x+1], maxpooled1[5][y-2][x+2],
                maxpooled1[5][y-1][x-2], maxpooled1[5][y-1][x-1], maxpooled1[5][y-1][x], maxpooled1[5][y-1][x+1], maxpooled1[5][y-1][x+2],
                  maxpooled1[5][y][x-2],   maxpooled1[5][y][x-1],   maxpooled1[5][y][x],   maxpooled1[5][y][x+1],   maxpooled1[5][y][x+2],
                maxpooled1[5][y+1][x-2], maxpooled1[5][y+1][x-1], maxpooled1[5][y+1][x], maxpooled1[5][y+1][x+1], maxpooled1[5][y+1][x+2],
                maxpooled1[5][y+2][x-2], maxpooled1[5][y+2][x-1], maxpooled1[5][y+2][x], maxpooled1[5][y+2][x+1], maxpooled1[5][y+2][x+2],
            ])
            for i in range(0, 6):
                    rows[i].append(max(0, np.sum(np.multiply(receptivefield, convmodel2[i]))))
        for h in range(0, 6):
            featuremap2[h].append(rows[h])
    featuremap2 = np.asarray(featuremap2)


    """maxpooling toepassen op alle 16 featuremaps"""
    maxpooled2 = [ [], [], [], [], [], [] ]
    for z in range(0, 6):
        collumn = []
        for y in range(0, 10, 2):
            row = []
            for x in range(0, 10, 2):
                row.append(max([featuremap2[z][y][x], featuremap2[z][y][x+1], featuremap2[z][y+1][x], featuremap2[z][y+1][x+1]]))
            collumn.append(row)
        maxpooled2[z] = np.asarray(collumn)
    maxpooled2 = np.asarray(maxpooled2)


    """dropout 2 toepassen"""
    if(training):
        maxpooled2.flags.writeable = True
        for z in range(0, 6):
            for y in range(0, 5):
                for x in range(0, 5):
                    if np.random.rand() < 0.05 :
                        maxpooled2[z][y][x] = 0


    """3 dimensies terugbrengen naar 1 dimensie"""
    flattened = maxpooled2.reshape(150)

    """fully connected layer 1"""
    fullyconnected1 = []


benchmark(100)
