import numpy as np
with open('data180k/train_list.txt','r') as f:
    lines = f.readlines()
series = [np.loadtxt(line.split()[0]) for line in lines]
series = np.array(series)
np.save('data180k/serieset.npy', series)
print(len(series))