import random
import numpy as np
import torch
from PIL import Image

class GenerateInput(object):
    def __init__(self, batch_size, train_lines, indexs, series_length, image_size_lab, train=True, ifstop=False, finetune=False, dataset='random', noise=0):
        
        self.batch_size = batch_size
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.series_length = series_length
        self.train = train
        self.finetune = finetune
        self.stop = ifstop
        self.iw,self.ih = image_size_lab
        self.noise = noise
        self.dataset = dataset
        if self.dataset == 'hadamard':
            self.H = np.load('data180k/hadamard_rank.npy')
        if self.train and not finetune:
            self.imageset = np.reshape(np.load('data180k/imageset.npy')[indexs], [len(indexs),64,64])
            if dataset == 'random':
                self.serieset = np.load('data180k/serieset.npy')[indexs,:series_length]
            elif dataset == 'hadamard':
                self.serieset = np.load('data180k/hadarank.npy')[indexs,:series_length]
            elif dataset == 'russian':
                self.serieset = np.load('data180k/russet.npy')[indexs,:series_length]
            elif dataset == 'fashion':
                self.imageset = np.reshape(np.load('Fashion_MNIST_train/fashionset.npy')[indexs], [len(indexs),64,64])
                self.serieset = self.imageset
            elif dataset == 'fashion_rd':
                self.imageset = np.reshape(np.load('Fashion_MNIST_train/fashionset.npy')[indexs], [len(indexs),64,64])
                self.serieset = np.load('Fashion_MNIST_train/fashionrdset.npy')[indexs,:series_length]
            else:
                self.serieset = np.load('data180k/%s.npy'%(dataset))[indexs,:series_length]
            self.serieset = np.array([(x - x.min())/(x.max()-x.min()) for x in self.serieset])
            self.boxset = []
            for line in self.train_lines:
                box = np.array([np.array(list(map(float,box.split(',')))) for box in line.split()[1:]], dtype=np.float32)
                if len(box)>0:
                    np.random.shuffle(box)
                    box_w = box[:, 2] + 0
                    box_h = box[:, 3] + 0
                    box[:,0] = box[:,0] / self.iw
                    box[:,2] = box[:,2] / self.iw
                    box[:,1] = box[:,1] / self.ih
                    box[:,3] = box[:,3] / self.ih
                    box[:, 0:2][box[:, 0:2]<0] = 0
                    box[:, 2][box[:, 2]>1] = 1
                    box[:, 3][box[:, 3]>1] = 1
                    box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
                    if(box[:,:4]==0).all():
                        box = []
                else:
                    box = []
                self.boxset.append(box)

    def generate(self):
        if self.train and not self.finetune:
            while True:
                indexs = np.array(range(len(self.imageset)))
                random.shuffle(indexs)
                self.imageset = np.array([self.imageset[x] for x in indexs])
                self.serieset = np.array([self.serieset[x] for x in indexs])
                self.boxset = np.array([self.boxset[x] for x in indexs])
                for itr in range(self.train_batches//self.batch_size):
                    tmp_inp = torch.FloatTensor(self.serieset[itr*self.batch_size:(itr+1)*self.batch_size])
                    if self.noise:
                        tmp_inp = torch.FloatTensor(tmp_inp+np.random.normal(0,self.noise,tmp_inp.shape))
                    tmp_targets = np.array(self.boxset[itr*self.batch_size:(itr+1)*self.batch_size])
                    tmp_img = torch.FloatTensor(self.imageset[itr*self.batch_size:(itr+1)*self.batch_size])
                    yield tmp_inp, tmp_targets, tmp_img
        
        print('Tips: test!')
        inputs = []
        targets = []
        images = []
        while True:
            random.shuffle(self.train_lines)
            lines = self.train_lines
            
            for iter, line in enumerate(lines):
                line = line.split()

                if self.dataset == 'fashion':
                    name = line[0]
                    series = Image.open(name).convert('L')
                    series = np.array(series, dtype=np.float32)/255.0
                else:
                    if self.dataset == 'fashion_rd':
                        series = np.loadtxt(line[0].replace('fig', 'txt').replace('png', 'txt'))[:self.series_length]
                    else:
                        series = np.loadtxt(line[0])[:self.series_length]
                    # Normalize
                    series = (series - series.min())/(series.max() - series.min())
                    #series = (series-series.mean())/series.max()

                    image = None
                if self.train:
                    name = line[0].replace('.txt', '.png').replace('jects/', 'jects_fig/').replace('ject/', 'ject_fig/').replace('selfmadetxt', 'selfmadeimg')
                    image = Image.open(name).convert('L')
                    image = np.array(image, dtype=np.float32)/255.0
                else:
                    image = line[0].split('/')[-1].split('.')[0].replace('test','test_fig')
                
                box = np.array([np.array(list(map(float,box.split(',')))) for box in line[1:]], dtype=np.float32)

                if self.dataset == 'TA':
                    name = line[0].replace('.txt', '.png').replace('jects/', 'jects_fig/').replace('ject/', 'ject_fig/').replace('selfmadetxt', 'selfmadeimg'.replace('test','test_fig'))
                    series = Image.open(name).convert('L')
                    series = np.array(series, dtype=np.float32)/255.0
                elif self.dataset == 'hadamard':
                    name = line[0].replace('.txt', '.png').replace('jects/', 'jects_fig/').replace('ject/', 'ject_fig/').replace('selfmadetxt', 'selfmadeimg'.replace('test','test_fig'))
                    series = Image.open(name).convert('L')
                    series = np.array(series, dtype=np.float32)/255.0
                    series = np.dot(np.reshape(series,[4096]), self.H.transpose())
                    series = (series - series.min())/(series.max() - series.min())
                
                if len(box)>0:
                    np.random.shuffle(box)
                    box_w = box[:, 2] + 0
                    box_h = box[:, 3] + 0
                    box[:,0] = box[:,0] / self.iw
                    box[:,2] = box[:,2] / self.iw
                    box[:,1] = box[:,1] / self.ih
                    box[:,3] = box[:,3] / self.ih
                    box[:, 0:2][box[:, 0:2]<0] = 0
                    box[:, 2][box[:, 2]>1] = 1
                    box[:, 3][box[:, 3]>1] = 1
                    box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
                    if(box[:,:4]==0).all():
                        box = []
                else:
                    box = []
                inputs.append(series)
                targets.append(box)
                images.append(image)

                if len(targets) == self.batch_size:
                    tmp_inp = torch.FloatTensor(inputs)
                    tmp_targets = np.array(targets)
                    tmp_img = torch.FloatTensor(images) if self.train else images
                    inputs = []
                    targets = []
                    images = []
                    yield tmp_inp, tmp_targets, tmp_img
            if self.stop:
                break

class FTLoader(object):
    def __init__(self, batch_size, train_lines, M, train=True):
        self.batchsize = batch_size
        self.trainlines = train_lines
        self.train = train
        self.M = M

    def generate(self):
        inputs = []
        outputs = []
        while self.train:
            random.shuffle(self.trainlines)
            for line in self.trainlines:
                line = line.split()
                series = np.loadtxt(line[0])[:self.M]
                refs = np.loadtxt(line[0].replace('Object/', 'Object_ref/').replace('Objects/', 'Objects_ref/'))[:self.M]
                series = (series - series.min()) / (series.max() - series.min())
                refs = (refs - refs.min()) / (refs.max() - refs.min())
                refs = refs - series
                inputs.append(series)
                outputs.append(refs)
                if len(outputs) == self.batchsize:
                    tmp1 = torch.FloatTensor(inputs)
                    tmp2 = torch.FloatTensor(outputs)
                    inputs = []
                    outputs = []
                    yield tmp1, tmp2