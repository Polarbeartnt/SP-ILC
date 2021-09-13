#coding = utf-8
#-------------------------------------#
#-------------------------------------#
from fashpredictor import Predictor
from utils.utils import Precision_Recall
import numpy as np
from mkdir import mkdir
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from netsn.network import FTNet
from netsn.network import MyNetBody as SimpleNet
from netsn.network import TrainANet
import csv
import random

PREDICT = True # whether need to predict, or just run evaluate
EVALUATE = False # whether need to evaluate, or predict only
# logs/bestnets_ex/Epoch40-Total_Loss24.2159-Val_Loss27.7022.pth
# logs/Wed_Jul_21_13:24:50_2021/Epoch18-Total_Loss17.6771-Val_Loss17.3474.pth
predictor = Predictor(_defaults = {
            "model_path"        : 'logs/Fri_Sep_10_10:33:10_2021/Epoch40-Total_Loss9.3831-Val_Loss9.3912.pth',#path of your weights-file
            "anchors_path"      : 'model_data/yolo_anchors16.txt',
            "classes_path"      : 'model_data/voc_classes.txt',
            "model_image_size"  : (64,64),
            "series_length"     : 333,#length of the S_array. Need to match your weights-file
            "input_to_net_size" : (512,512),
            "confidence"        : 0.7,# confidence threshold can be modified here 
            "iou"               : 0.5,
            "cuda"              : False,
            "MyNetBody"         : TrainANet, # which net used
            "FineTune"          : None,
            "FT_log"            : '',
            "dataset"           : 'fashion'
        })

Path = ['fash']
for path in Path:
    mkdir('test_results', path)
    f = open('test_results/results_%s.txt'%(path), 'a+')
    f.write(predictor.model_path+'\n')
    with open('fashion0909/test_%s.txt'%(path), 'r') as g:
        exfile = [tmp.split(' ')[0] for tmp in g.readlines()]
    random.seed(6708)
    random.shuffle(exfile)
    random.seed(None)
    exfile = exfile[:100]
    if PREDICT:
        predictor.get_detect_results(exfile,path)
    print('Start Evaluatine......')
    psnr = []
    ssim = []
    for p in exfile:
        p = p.replace('\n', '')
        name = p.split('/')[-1]
        fig = np.array(Image.open(p.replace('Objects', 'Objects_fig').replace('Object/', 'Object_fig/').replace('.txt', '.png')).convert('L'))
        rec = np.array(Image.open('test_results/' + path + '_predict/'+name.replace('.txt', '.png')).convert('L'))
        psnr.append(peak_signal_noise_ratio(fig,rec,data_range=255))
        ssim.append(structural_similarity(fig,rec,data_range=255))
        
        print('Finished: ', p)
    np.savetxt('test_results/'+path+'_psnr.txt', np.array(psnr))
    np.savetxt('test_results/'+path+'_ssim.txt', np.array(ssim))
    f.write(path+'\n')
    f.write('psnr: %s \n'%(np.mean(psnr)))
    f.write('ssim: %s \n'%(np.mean(ssim)))
    f.close()