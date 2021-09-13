import numpy as np

with open('fashion0909/train_fash.txt','r') as f:
    lines = f.readlines()
names = [line.split()[0].replace('fig', 'txt').replace('.png', '.txt') for line in lines]
s = []
l = len(names)
for i,line in enumerate(names):
    arrau = np.loadtxt(line)
    s.append(arrau)
s = np.array(s)
np.save('fashion0909/fashionrdset.npy', s)
print(len(s))