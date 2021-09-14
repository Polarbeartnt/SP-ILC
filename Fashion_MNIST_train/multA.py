import numpy as np
from PIL import Image

with open('Fashion_MNIST_train/train_fash.txt', 'r') as f:
    filelist = f.readlines()

A = np.loadtxt('A_8192.txt')
A[A==-1] = 0

tot = len(filelist)
totseries = []
for ord, name in enumerate(filelist):
    name = name.split(' ')[0]
    fig = Image.open(name)
    figarray = np.reshape(np.array(fig), [4096,])
    s = np.dot(figarray, A.transpose())
    totseries.append(s)
    np.savetxt(name.replace('fig', 'txt').replace('png', 'txt'), s)
    print('Done %s / %s.'%(ord,tot))

totseries = np.array(totseries)
np.save('Fashion_MNIST_train/randomfash.npy', totseries)