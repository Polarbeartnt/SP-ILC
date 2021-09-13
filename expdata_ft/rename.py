import os

root = 'expdata_ft/Multiple_Objects'
path = ['/', '_fig/', '_ref/']

for p in path:
    names = os.listdir(root+p)
    for name in names:
        newname = '%03d.txt'%(int(name.split('.')[0])+139)
        os.rename(root+p+name, root+p+newname)

for name in os.listdir(root+path[1]):
    os.rename(root+path[1]+name, root+path[1]+name.replace('txt','png'))