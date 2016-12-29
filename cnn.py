import numpy as np
from PIL import Image
import time


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


model = np.asarray([

   [np.asarray(np.random.rand(25)),
    np.asarray(np.random.rand(25)),
    np.asarray(np.random.rand(25)),
    np.asarray(np.random.rand(25)),
    np.asarray(np.random.rand(25)),
    np.asarray(np.random.rand(25))],


   [np.asarray(np.random.rand(6,25)),
    np.asarray(np.random.rand(6,25)),
    np.asarray(np.random.rand(6,25)),
    np.asarray(np.random.rand(6,25)),
    np.asarray(np.random.rand(6,25)),
    np.asarray(np.random.rand(6,25)),
    np.asarray(np.random.rand(6,25)),
    np.asarray(np.random.rand(6,25)),
    np.asarray(np.random.rand(6,25)),
    np.asarray(np.random.rand(6,25)),
    np.asarray(np.random.rand(6,25)),
    np.asarray(np.random.rand(6,25)),
    np.asarray(np.random.rand(6,25)),
    np.asarray(np.random.rand(6,25)),
    np.asarray(np.random.rand(6,25)),
    np.asarray(np.random.rand(6,25))]])


def cnn(imagePath, model):

    starttime = time.time()

    """plaatje openen"""
    img = np.asarray(Image.open(imagePath))

    """3d plaatje omzetten naar een 2d array met alleen de kleurintensiteit (zwart-wit)"""
    input = []
    for y in xrange(0, 32):
        row = []
        for x in xrange(0, 32):
            row.append(max(img[y][x]))
        input.append(row)
    inputlayer = np.asarray(input)

    """eerste featuremaps creeeren. activatiefuncties worden ook meteen toegepast"""
    featuremap1 = [ [], [], [], [], [], [] ]
    convmodel1 = model[0]
    for y in xrange(2, 30):
        row1 = []
        row2 = []
        row3 = []
        row4 = []
        row5 = []
        row6 = []
        for x in xrange(2, 30):
            receptivefield = np.asarray([
                inputlayer[y-2][x-2], inputlayer[y-2][x-1], inputlayer[y-2][x], inputlayer[y-2][x+1], inputlayer[y-2][x+2],
                inputlayer[y-1][x-2], inputlayer[y-1][x-1], inputlayer[y-1][x], inputlayer[y-1][x+1], inputlayer[y-1][x+2],
                  inputlayer[y][x-2],   inputlayer[y][x-1],   inputlayer[y][x],   inputlayer[y][x+1], inputlayer[y][x+2],
                inputlayer[y+1][x-2], inputlayer[y+1][x-1], inputlayer[y+1][x], inputlayer[y+1][x+1], inputlayer[y+1][x+2],
                inputlayer[y+2][x-2], inputlayer[y+2][x-1], inputlayer[y+2][x], inputlayer[y+2][x+1], inputlayer[y+2][x+2],
            ])
            row1.append(max(0, np.sum(np.multiply(receptivefield, convmodel1[0]))))
            row2.append(max(0, np.sum(np.multiply(receptivefield, convmodel1[1]))))
            row3.append(max(0, np.sum(np.multiply(receptivefield, convmodel1[2]))))
            row4.append(max(0, np.sum(np.multiply(receptivefield, convmodel1[3]))))
            row5.append(max(0, np.sum(np.multiply(receptivefield, convmodel1[4]))))
            row6.append(max(0, np.sum(np.multiply(receptivefield, convmodel1[5]))))
        featuremap1[0].append(row1)
        featuremap1[1].append(row2)
        featuremap1[2].append(row3)
        featuremap1[3].append(row4)
        featuremap1[4].append(row5)
        featuremap1[5].append(row6)
    featuremap1 = np.asarray(featuremap1)


    """maxpooling toepassen op alle 6 featuremaps"""
    maxpooled1 = [ [], [], [], [], [], [] ]
    for z in xrange(0, 6):
        collumn = []
        for y in xrange(0, 28, 2):
            row = []
            for x in xrange(0, 28, 2):
                row.append(max([featuremap1[z][y][x], featuremap1[z][y][x+1], featuremap1[z][y+1][x], featuremap1[z][y+1][x+1]]))
            collumn.append(row)
        maxpooled1[z] = np.asarray(collumn)
    maxpooled1 = np.asarray(maxpooled1)


    """tweede featuremaps creeeren, activatiefuncties worden ook meteen toegepast"""
    for y in xrange(2, 12):
        for x in xrange(2, 12):
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


    print "--------------------------------------------------------\nhet convnet heeft er " + str(time.time() - starttime)+ "s over gedaan"
    print maxpooled1.shape


cnn('dataset/0/img001-001.png', model)
