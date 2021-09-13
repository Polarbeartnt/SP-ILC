#coding = utf-8
#-------------------------------------#
#-------------------------------------#
from predictor import Predictor
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

PREDICT = True # whether need to predict, or just run evaluate
EVALUATE = True # whether need to evaluate, or predict only
# logs/bestnets_ex/Epoch40-Total_Loss24.2159-Val_Loss27.7022.pth
# logs/Wed_Jul_21_13:24:50_2021/Epoch18-Total_Loss17.6771-Val_Loss17.3474.pth
predictor = Predictor(_defaults = {
            "model_path"        : 'logs/训练A矩阵/Epoch40-Total_Loss8.4614-Val_Loss11.1676.pth',#path of your weights-file
            "anchors_path"      : 'model_data/yolo_anchors16.txt',
            "classes_path"      : 'model_data/voc_classes.txt',
            "model_image_size"  : (64,64),
            "series_length"     : 333,#length of the S_array. Need to match your weights-file
            "input_to_net_size" : (512,512),
            "confidence"        : 0.976,# confidence threshold can be modified here 
            "iou"               : 0.5,
            "cuda"              : False,
            "MyNetBody"         : TrainANet, # which net used
            "FineTune"          : None,
            "FT_log"            : '',
            "dataset"           : 'TA'
        })

Path = ['Single_Object', 'Multiple_Objects', 'test']
for path in Path:
    mkdir('test_results', path)
    f = open('test_results/results_%s.txt'%(path), 'a+')
    f.write(predictor.model_path+'\n')
    with open('expdata_ft/test_%s.txt'%(path), 'r') as g:
        exfile = [tmp.split(' ')[0] for tmp in g.readlines()]
    if PREDICT:
        predictor.get_detect_results(exfile,path)
    if not EVALUATE:
        continue
    print('Start Evaluatine......')
    psnr = []
    ssim = []
    box_eval = []
    for p in exfile:
        name = p.split('/')[-1]
        fig = np.array(Image.open(p.replace('Objects', 'Objects_fig').replace('Object/', 'Object_fig/').replace('.txt', '.png')).convert('L'))
        rec = np.array(Image.open('test_results/' + path + '_predict/'+name.replace('.txt', '.png')).convert('L'))
        pr = Precision_Recall('test_results/'+path+'_bbox/'+name, p.replace('Objects', 'Objects_label').replace('Object/', 'Object_label/'),confbar=[0.9999])
        psnr.append(peak_signal_noise_ratio(fig,rec,data_range=255))
        ssim.append(structural_similarity(fig,rec,data_range=255))
        box_eval.append(pr)
        print('Finished: ', p)
    np.savetxt('test_results/'+path+'_psnr.txt', np.array(psnr))
    np.savetxt('test_results/'+path+'_ssim.txt', np.array(ssim))
    f.write(path+'\n')
    f.write('psnr: %s \n'%(np.mean(psnr)))
    f.write('ssim: %s \n'%(np.mean(ssim)))
    num = len(box_eval)
    box_eval = np.sum(np.array(box_eval), axis=0)
    recall = box_eval[:, 0] / box_eval[:, 1]
    precision = box_eval[:, 2] / box_eval[:, 3]
    confidence = box_eval[:,4] / num
    f.write('find:%s, total:%s, Recall:%s\n'%(box_eval[0, 0], box_eval[0, 1], recall[0]))
    f.write('correct:%s, predict:%s, Precision:%s\n'%(box_eval[0, 2], box_eval[0, 3], precision[0]))
    f.write('confidence:%s'%(predictor.confidence))
    f.close()

    with open('test_results/pr.csv', 'a+') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(recall)
        f_csv.writerow(precision)
        f_csv.writerow(confidence)

    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('test_results/pr.png')

Path = ['double_mnist', 'triple_mnist', 'dt']
mkdir('test_results', Path[0])
prf = open('test_results/pr.csv', 'a+')
for path in Path:
    f = open('test_results/results_%s.txt'%(path), 'a+')
    mkdir('test_results', path)
    f.write(predictor.model_path+'\n')
    with open('DouTriMnist/test_%s.txt'%(path), 'r') as g:
        exfile = [tmp.split(' ')[0] for tmp in g.readlines()]
    if PREDICT:
        predictor.get_detect_results(exfile,path)
    if not EVALUATE:
        continue
    print('Start Evaluatine......')
    psnr = []
    ssim = []
    box_eval = []
    for p in exfile:
        p = p.split('\n')[0]
        name = p.split('/')[-1]
        fig = np.array(Image.open(p.replace('test', 'test_fig').replace('.txt', '.png')).convert('L'))
        rec = np.array(Image.open('test_results/' + path + '_predict/'+name.replace('.txt', '.png')).convert('L'))
        pr = Precision_Recall('test_results/'+path+'_bbox/'+name, p.replace('test', 'test_label'),confbar=[0.9999])
        if np.sum(pr,axis=0)[2] < np.sum(pr,axis=0)[3]:
            with open('error.txt', 'a+') as err:
                err.write(p+'\n')
        psnr.append(peak_signal_noise_ratio(fig,rec,data_range=255))
        ssim.append(structural_similarity(fig,rec,data_range=255))
        box_eval.append(pr)
        print('Finished: ', p)
    np.savetxt('test_results/'+path+'_psnr.txt', np.array(psnr))
    np.savetxt('test_results/'+path+'_ssim.txt', np.array(ssim))
    f.write(path+'\n')
    f.write('psnr: %s \n'%(np.mean(psnr)))
    f.write('ssim: %s \n'%(np.mean(ssim)))
    num = len(box_eval)
    box_eval = np.sum(np.array(box_eval), axis=0)
    recall = box_eval[:, 0] / box_eval[:, 1]
    precision = box_eval[:, 2] / box_eval[:, 3]
    confidence = box_eval[:,4] / num
    f.write('find:%s, total:%s, Recall:%s\n'%(box_eval[0, 0], box_eval[0, 1], recall[0]))
    f.write('correct:%s, predict:%s, Precision:%s\n'%(box_eval[0, 2], box_eval[0, 3], precision[0]))
    f.write('confidence:%s'%(predictor.confidence))
    f.close()

    f_csv = csv.writer(prf)
    f_csv.writerow(recall)
    f_csv.writerow(precision)
    f_csv.writerow(confidence)

    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('test_results/pr_%s.png'%(path))

f.close()