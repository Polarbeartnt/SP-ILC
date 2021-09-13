import cv2,os
import random

root = 'dataset/double_mnist/test'
files = [x for x in os.listdir(root) if x.find('png') != -1]
for name in files:
    fname = '/'.join([root,name])

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not os.path.exists(root+'_label/'):
        os.mkdir(root+'_label/')
    error = False
    with open(root+'_label/'+name.replace('png','txt'), 'w') as f:
        for i,ct in enumerate(contours):
            x1 = str(min(ct[:,0,0])*512//64)
            x2 = str(max(ct[:,0,0])*512//64)
            y1 = str(min(ct[:,0,1])*512//64)
            y2 = str(max(ct[:,0,1])*512//64)
            try:
                c = str(name.split('.')[0][i])
            except:
                error = True
                c = 'None'
            f.write(' '.join([x1,y1,x2,y2,c])+'\n')
    if error:
        os.remove(root+'_label/'+name.replace('png','txt'))
        os.remove(fname)
        print('Error occurred, delete '+root+'_label/'+name.replace('png','txt'))

random.seed(123)
files = [x for x in os.listdir(root) if x.find('png') != -1]
random.shuffle(files)
random.seed(None)
for name in files[20:]:
    fname = '/'.join([root,name])
    os.remove(root+'_label/'+name.replace('png','txt'))
    os.remove(fname)