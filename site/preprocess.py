import numpy as np
from PIL import Image
import sys

imgPath = sys.argv[1]
img = Image.open(imgPath)
img.thumbnail((32, 32), Image.ANTIALIAS)
container = Image.new('RGB', (32, 32), (255, 255, 255))
container.paste(
img, (int((32 - img.size[0]) / 2), int((32 - img.size[1]) / 2))
)
container.save(imgPath)
