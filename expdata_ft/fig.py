import numpy as np
from PIL import Image
import os

data = 'Single_Object'
if not os.path.exists(data+'_fig'):
    os.mkdir(data+'_fig')
A = np.loadtxt('../A_8192.txt')
iA = np.linalg.inv(A)

sets = os.listdir(data+'_ref')
for name in sets:
    s = np.loadtxt(data+'_ref/'+name)
    fig = np.dot(iA,s)
    fig = np.reshape(fig, [64,64])
    fig[fig<0] = 0
    fig[fig>255] = 255
    fig = Image.fromarray(fig).convert('RGB').save(data+'_fig/'+name.replace('txt', 'png'))
