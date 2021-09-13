import numpy as np
import os
from PIL import Image

MatrixA = np.loadtxt('../A_8192.txt')

root = 'dataset/double_mnist/test_fig'
new_root = 'dataset/double_mnist/test'

if not os.path.exists(new_root):
    os.mkdir(new_root)

if True:
    images = os.listdir('%s'%(root))
    tot = len(images)
    for imgpath in images:
        fp = open('%s/%s'%(root,imgpath), 'rb')
        img = Image.open(fp).convert('L')
        fp.close()
        img = np.array(img)
        s = np.dot(np.reshape(img,[img.shape[0]*img.shape[1]]),np.transpose(MatrixA))
        imgname = imgpath.split('.')[0] + '.txt'
        np.savetxt('%s/%s'%(new_root,imgname), s)
