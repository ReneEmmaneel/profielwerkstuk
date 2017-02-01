import numpy as np
import cnn as cnn
import math as math
import random as random
import time as time
import os.path
import random

#training functie
def train(alpha, totalGeneration, modelfile, metadatafile):

    #zorgen dat data goed wordt weggeschreven
    with open(metadatafile, "w") as myfile:
        myfile.write('\n\n\n\n')

    #tijd en generatie bijhouden
    startTime = int(time.time())

    vervangTekst(metadatafile, 0, "Training gestart op " + str(int(time.time())) + "..\n")

    #alle labels
    lettersArray = ['0','1']

    #vectors voor classificatie systeem genereren
    AllLabels = np.zeros((len(lettersArray), len(lettersArray)))
    for x in xrange(len(lettersArray)):
        for y in xrange(len(lettersArray)):
            if x == y:
                AllLabels[x][y] = 1

    #het maken van de startweights
    hiddenSize = 150
    #hiddenSize zijn het aantal nodes aan het einde van de 2e hidden layer
    model = np.asarray([
        
        np.asarray([[-1, 0, 2, 0, -1, 0, -1, 3, -1, 0, 2, 3, 4, 3, 2, 0, -1, 3, -1, 0, -1, 0, 2, 0, -1],
         [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 3, -5, 4, 1, 0, -1, 3, -1, 0, 0, -2, 1, -2, 0],
         [0, -1, 3, -1, 0, 0, -1, 3, -1, 0, 0, -1, 3, -1, 0, 0, -1, 3, -1, 0, 0, -1, 3, -1, 0],
         [0, 0, 0, 0, 0, -1, -1, -1, -1, -1, 3, 3, 3, 3, 3, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0],
         [0, -2, 1, -2, 0, 0, -1, 3, -1, 0, 1, 3, -5, 3, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
         [2, 1, -1, 1, 2, 1, 3, 1, 3, 1, -1, 1, 4, 1, -1, 1, 3, 1, 3, 1, 2, 1, -1, 1, 2]]),

        np.asarray([[1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 4, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 4, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 4, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 4, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 4, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 4, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1],
         [4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 2, 0, 2, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 2, 0, 2, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 2, 0, 2, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 2, 0, 2, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 2, 0, 2, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 2, 0, 2, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4],
         [0, 0, 0, 0, 0, -1, -1, 1, -1, -1, 1, 1, 2, 1, 1, -1, -1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 4, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, -2, -2, -2, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 1, 2, 0, 1, 1, 2, 1, 1, 0, 2, 1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 4, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 2, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 1, 2, 1, 0, 1, 2, 4, 2, 1, 0, 1, 2, 1, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, -1, -1, -1, 0, -1, -1, -2, -1, -1, 0, -1, -1, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, -1, 0, 0, 0, -1, -1, 0, 1, 0, -1, -1, 0, 0, 0, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, -1, -1, 0, -1, -1, -2, -1, -1, 0, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 2, 1, 2, 0, 1, 1, 4, 1, 1, 0, 2, 1, 2, 0, 0, 0, 1, 0, 0],
         [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 4, 2, 0, 2, 4, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 3, 2, 0, 1, 3, 4, 3, 1, 0, 2, 3, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, -1, -2, -1, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 2, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -2, -3, -2, 0, -1, -3, -4, -3, -1, 0, -2, -3, -2, 0, 0, 0, -1, 0, 0, 0, -1, 1, 3, 4, -1, -1, 1, 2, 3, 1, 1, 1, 1, 1, 3, 2, 1, -1, -1, 4, 3, 1, -1, 0],
         [0, 0, 2, 0, 0, 0, 1, 2, 1, 0, 2, 2, 3, 2, 2, 0, 1, 2, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, -1, -2, -1, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 3, 1, 0, 0, 1, 4, 1, 0, 0, 1, 3, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 2, 3, 4, 3, 2, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, -1, -2, -1, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 3, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
        ]),
        
        [np.asarray(np.random.rand((hiddenSize + 1) * len(lettersArray)))]
        
    ])

    #importen van data
    labelSet = []
    dataSet = []

    for x in xrange(len(lettersArray)):
        y = 0
        while os.path.isfile("dataset/" + lettersArray[x] + "/img" + str(x + 1).zfill(3)  + "-" + str(y + 1).zfill(3) + ".png"):
            dataSet.append(str("dataset/" + lettersArray[x] + "/img" + str(x + 1).zfill(3)  + "-" + str(y + 1).zfill(3) + ".png"))
            labelSet.append(int(x))
            y += 1
        y = 0
        while os.path.isfile("dataset/" + lettersArray[x] + "/img" + str(x + 1).zfill(3)  + "-" + str(y + 1).zfill(5) + ".png"):
            dataSet.append(str("dataset/" + lettersArray[x] + "/img" + str(x + 1).zfill(3)  + "-" + str(y + 1).zfill(5) + ".png"))
            labelSet.append(int(x))
            y += 1
    dataSet = np.asarray(dataSet)

    #label koppelen aan data
    allData = []
    i = 0
    for data in dataSet:
        allData.append([data, labelSet[i]])
        i += 1

    #start van nieuwe generatie
    for numLoop in xrange(totalGeneration):
        #metadatafile aanpassen
        vervangTekst(metadatafile, 1, "Bezig met generatie " + str(numLoop) + "..\n")

        #willekeurige data kiezen
        gebruikteData = allData[random.randint(0, len(allData) - 1)]
        print gebruikteData[0]
        
        #neurale netwerk uitvoeren
        outputData = cnn.cnn(gebruikteData[0], model, AllLabels[gebruikteData[1]], True)

        #backpropagation
        for i in range(len(model[2][0])):
            x = i%(hiddenSize + 1)
            y = math.floor(i/(hiddenSize + 1))
            derivitive = (outputData[2][y] - outputData[1][y]) * outputData[3][x]
            model[2][0][i] -= alpha * derivitive
        
        #gegevens in de documenten aanpassen
        with open(modelfile, "w") as myfile:
            myfile.write(str(model[0]) + ',\n')
            myfile.write(str(model[1]) + ',\n')
            myfile.write(str(np.asarray(model[2])) + ',\n')
            
        with open(metadatafile, "a") as myfile:
            myfile.write(str(numLoop) + ' ' + str(outputData[1][1]) + ' ' + str(outputData[0]) + ' ' + str(outputData[2][0]) + '\n')

def vervangTekst(file, line, text):
    lines = open(file, 'r').readlines()
    lines[line] = text
    out = open(file, 'w')
    out.writelines(lines)
    out.close()