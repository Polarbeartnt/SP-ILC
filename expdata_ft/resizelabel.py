import os

pathlist = ['expdata/Single_Object_label/', 'expdata/Multiple_Objects_label/']
for path in pathlist:
    files = os.listdir(path)
    for name in files:
        name = path + name
        with open(name, 'r') as f:
            labels = f.readlines()
        newlabel = []
        for line in labels:
            newlabel.append(' '.join([str(int(int(l) * 512 / 416)) for l in line.split(' ')[:4]] + [line.split(' ')[4]]))
        with open(name, 'w') as f:
            f.writelines(newlabel)
        