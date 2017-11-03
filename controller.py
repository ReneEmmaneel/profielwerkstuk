import cnn as cnn
import training as training
import time
import numpy

numpy.set_printoptions(threshold=numpy.nan)
training.train(0.00000001, 10, "model.doc", "metadata.doc")