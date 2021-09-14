import os
import numpy as np

def check(l):
    classs = l[4]
    l = np.array(l[:4],dtype=np.float32)
    x1 = str(int((l[0]-l[2]//2)*512/64))
    y1 = str(int((l[1]-l[3]//2)*512/64))
    x2 = str(int((l[0]+l[2]//2)*512/64))
    y2 = str(int((l[0]+l[3]//2)*512/64))
    return ' '.join([x1,y1,x2,y2,classs])

root = 'Fashion_MNIST_test/'
files = [root+'label/'+name for name in os.listdir(root+'label')]

for name in files:
    with open(name, 'r') as f:
        label = [check(l.split(' ')) for l in f.readlines()]
    with open(name.replace('label', 'label_test'), 'w') as f:
        f.writelines(label)
