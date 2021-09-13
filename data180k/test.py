import numpy as np
from PIL import Image
a = np.load('data180k/imageset.npy')[5923]*255
b = np.loadtxt('data180k/selfmadetxt/0/00005924.txt')
A = np.loadtxt('A_8192.txt')
print(np.dot(A,a)-b)