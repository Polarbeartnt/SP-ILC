import numpy as np
from scipy.linalg import hadamard
import cv2,os

#os.mkdir('data180k/hadarank')

#images = np.load('data180k/imageset.npy')
H = hadamard(16)
scores = []
for pattern in H:
    pattern = (np.reshape(pattern, [4,4])+1)*127
    pattern = np.array(pattern,dtype=np.uint8)
    pattern = cv2.resize(pattern, (512,512), interpolation=cv2.INTER_NEAREST)
    ret, binary = cv2.threshold(cv2.erode(pattern,np.ones([2,2],dtype=np.uint8)),127,255,cv2.THRESH_BINARY)
    ret, binary_inv = cv2.threshold(cv2.erode(255-pattern,np.ones([2,2],dtype=np.uint8)),127,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contours_inv, hierarchy = cv2.findContours(binary_inv,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contours += contours_inv
    scores.append(len(contours))

rank = np.argsort(scores)
print(rank)
H = H[rank]
cv2.imwrite('data180k/hadrank.png', (H+1)*127)