from PIL import Image
import numpy as np
from numpy.lib.type_check import imag

with open('Fashion_MNIST_train/train_fash.txt','r') as f:
    lines = f.readlines()
names = [line.split()[0].replace('.txt', '.png').replace('label', 'fig') for line in lines]
images = []
l = len(names)
for i,line in enumerate(names):
    image = Image.open(line).convert('L')
    image = np.array(image, dtype=np.float32)/255.0
    image = np.reshape(image, [4096])
    images.append(image)
images = np.array(images)
np.save('Fashion_MNIST_train/fashionset.npy', images)
print(len(images))