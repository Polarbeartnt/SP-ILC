import os, shutil
import cv2

root = 'dataset/double_mnist/test'
files = [x for x in os.listdir(root) if x.find('png') == -1]
for name in files:
    fname = '/'.join([root,name,'0_'+name+'.png'])
    fig = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    ret, fig = cv2.threshold(fig,127,255,cv2.THRESH_BINARY)
    fig = cv2.resize(fig, (64,64))
    ret, fig = cv2.threshold(fig,127,255,cv2.THRESH_BINARY)
    
    cv2.imwrite('/'.join([root,name])+'.png',fig)
    shutil.rmtree('/'.join([root,name])+'/')