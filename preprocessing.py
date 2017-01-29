import numpy as np
from PIL import Image

def preprocess(imgPath):
    target = Image.open(imgPath);
