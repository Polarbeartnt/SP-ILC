import numpy as np
import cv2, os, random

def TestDataSet(filepath='fashion0909/fig'):
    filenames = os.listdir(filepath)

    imagenames = []
    for name in filenames:
        namelist = name.split('.')
        if namelist[-1] == 'png':
            imagenames.append(namelist[0])

    if not os.path.exists(filepath+'test'):
        os.mkdir(filepath+'test')
    
    random.shuffle(imagenames)
    for filename in imagenames[:100]:
        with open('{0}/{1}.txt'.format(filepath.replace('fig', 'label'), filename), 'r') as labelfile:
            lab = labelfile.read().split('\n')[:-1]
        image = cv2.imread('{0}/{1}.png'.format(filepath, filename))
        image = cv2.resize(image, (416,416), interpolation=cv2.INTER_NEAREST)
        for line in lab:
            label = [int(a) for a in line.split(',')]
            x1 = int(int(label[0] - label[2]//2) * 6.5)
            y1 = int(int(label[1] - label[3]//2) * 6.5)
            x2 = int(int(label[0] + label[2]//2) * 6.5)
            y2 = int(int(label[1] + label[3]//2) * 6.5)
            cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), thickness=2)
            cv2.rectangle(image, (x1,y1+13), (x1+170,y1), (0,255,0), thickness=cv2.FILLED)
            cv2.putText(image, 'label={0},x={1},y={2},w={3},h={4}'
                .format(label[4],label[0],label[1],label[2],label[3]), (x1,y1+13), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))
        cv2.imwrite('%stest/%s.png'%(filepath,filename), image)

    print('Data designed for test is available.')

if __name__ == '__main__':
    TestDataSet()