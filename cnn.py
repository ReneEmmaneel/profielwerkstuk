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

"""het door het cnn gebruikte model"""
model = np.asarray([

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
    np.asarray(np.random.rand(150)),
    np.asarray(np.random.rand(150)),
    np.asarray(np.random.rand(150)),
    np.asarray(np.random.rand(150)),
    np.asarray(np.random.rand(150)),
    np.asarray(np.random.rand(150)),
    np.asarray(np.random.rand(150)),
    np.asarray(np.random.rand(150)),
    np.asarray(np.random.rand(150)),
    np.asarray(np.random.rand(150)),
    np.asarray(np.random.rand(150))]
])


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
    featuremap2 = [ [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []  ]
    convmodel2 = model[1]
    for y in xrange(2, 12):
        row1 = []
        row2 = []
        row3 = []
        row4 = []
        row5 = []
        row6 = []
        row7 = []
        row8 = []
        row9 = []
        row10 = []
        row11 = []
        row12 = []
        row13 = []
        row14 = []
        row15 = []
        row16 = []
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
            row1.append(max(0, np.sum(np.multiply(receptivefield, convmodel2[0]))))
            row2.append(max(0, np.sum(np.multiply(receptivefield, convmodel2[1]))))
            row3.append(max(0, np.sum(np.multiply(receptivefield, convmodel2[2]))))
            row4.append(max(0, np.sum(np.multiply(receptivefield, convmodel2[3]))))
            row5.append(max(0, np.sum(np.multiply(receptivefield, convmodel2[4]))))
            row6.append(max(0, np.sum(np.multiply(receptivefield, convmodel2[5]))))
            row7.append(max(0, np.sum(np.multiply(receptivefield, convmodel2[6]))))
            row8.append(max(0, np.sum(np.multiply(receptivefield, convmodel2[7]))))
            row9.append(max(0, np.sum(np.multiply(receptivefield, convmodel2[8]))))
            row10.append(max(0, np.sum(np.multiply(receptivefield, convmodel2[9]))))
            row11.append(max(0, np.sum(np.multiply(receptivefield, convmodel2[10]))))
            row12.append(max(0, np.sum(np.multiply(receptivefield, convmodel2[11]))))
            row13.append(max(0, np.sum(np.multiply(receptivefield, convmodel2[12]))))
            row14.append(max(0, np.sum(np.multiply(receptivefield, convmodel2[13]))))
            row15.append(max(0, np.sum(np.multiply(receptivefield, convmodel2[14]))))
            row16.append(max(0, np.sum(np.multiply(receptivefield, convmodel2[15]))))
        featuremap2[0].append(row1)
        featuremap2[1].append(row2)
        featuremap2[2].append(row3)
        featuremap2[3].append(row4)
        featuremap2[4].append(row5)
        featuremap2[5].append(row6)
        featuremap2[6].append(row7)
        featuremap2[7].append(row8)
        featuremap2[8].append(row9)
        featuremap2[9].append(row10)
        featuremap2[10].append(row11)
        featuremap2[11].append(row12)
        featuremap2[12].append(row13)
        featuremap2[13].append(row14)
        featuremap2[14].append(row15)
        featuremap2[15].append(row16)
    featuremap2 = np.asarray(featuremap2)


    """maxpooling toepassen op alle 16 featuremaps"""
    maxpooled2 = [ [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []  ]
    for z in xrange(0, 16):
        collumn = []
        for y in xrange(0, 10, 2):
            row = []
            for x in xrange(0, 10, 2):
                row.append(max([featuremap2[z][y][x], featuremap2[z][y][x+1], featuremap2[z][y+1][x], featuremap2[z][y+1][x+1]]))
            collumn.append(row)
        maxpooled2[z] = np.asarray(collumn)
    maxpooled2 = np.asarray(maxpooled2)


    """3 dimensies terugbrengen naar 2 dimensies"""
    flattened = maxpooled2.reshape(80, 5)


    """fully connected layer 1"""
    fullyconnected1 = []

    print "--------------------------------------------------------\nhet convnet heeft er " + str(time.time() - starttime)+ "s over gedaan"



cnn('dataset/0/img001-001.png', model)
