import numpy as np
import os
from PIL import Image

MatrixA = np.loadtxt('../A_8192.txt')

root = 'selfmadeimg'
new_root = 'selfmadetxt'
folders = ['2']

if not os.path.exists(new_root):
    os.mkdir(new_root)

cnt = 0
for folder in folders:
    if not os.path.exists('%s/%s'%(new_root, folder)):
        os.mkdir('%s/%s'%(new_root, folder))
    images = os.listdir('%s/%s'%(root,folder))
    tot = len(folders)*len(images)
    for imgpath in images:
        fp = open('%s/%s/%s'%(root,folder,imgpath), 'rb')
        img = Image.open(fp).convert('L')
        fp.close()
        img = np.array(img)
        s = np.dot(np.reshape(img,[img.shape[0]*img.shape[1]]),np.transpose(MatrixA))
        imgname = imgpath.split('.')[0] + '.txt'
        np.savetxt('%s/%s/%s'%(new_root,folder,imgname), s)
        cnt += 1
        print('Finished %s / %s.'%(cnt, tot))
