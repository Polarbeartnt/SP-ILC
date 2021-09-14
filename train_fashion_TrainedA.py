#-------------------------------------#
#       Main for training
#-------------------------------------#
import os
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchsummary import summary
from netsn.loss import YOLOLoss, RestrictionLoss
from netsn.network import TrainANet
from tqdm import tqdm
from utils.dataloader import GenerateInput
#import profile
from PIL import Image
from multiprocessing import cpu_count

cpu_num = cpu_count() # 自动获取最大核心数目
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num-5)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1,3,2])

def fit_ont_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda):
    total_loss = 0
    loss_loc = 0
    loss_conf = 0
    loss_cls = 0
    loss_img = 0
    loss_bi = 0
    val_loss = 0
    val_loc = 0
    val_conf = 0
    val_cls = 0
    val_img = 0
    val_bi = 0
    start_time = time.time()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            s_series, targets, images = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    s_series = Variable(s_series).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets]
                    images = Variable(images).cuda()
                else:
                    s_series = Variable(s_series)
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                    images = Variable(images)
            optimizer.zero_grad()
            outputs = net(images)
            losses = []
            lloc,lconf,lcls = 0,0,0
            for i in range(2):
                loss_item = yolo_losses[i](outputs[i], targets)
                losses.append(loss_item[0])
                lloc += loss_item[3]
                lconf += loss_item[1]
                lcls += loss_item[2]
            reconstruct = outputs[2]
            weights = outputs[4]
            lbi = 0.01*binary(weights)
            limg = 1000*crit(reconstruct, images)
            losses.append(limg)
            losses.append(lbi)
            loss = sum(losses)
            loss.backward()
            optimizer.step()
            if iteration == 1:
                recs = reconstruct.cpu().detach().numpy()
                imgs = images.cpu().detach().numpy()
                forfig = outputs[3]
                forfig = forfig.cpu().detach().numpy()
                for i in range(1):
                    rec = Image.fromarray(recs[i]*255.0/recs[i].max()).convert('RGB')
                    img = Image.fromarray(imgs[i]*255.0).convert('RGB')
                    fig = Image.fromarray(forfig[i][0]*255.0/forfig[i][0].max()).convert('RGB')
                    rec.save('%s/%s/%s_%srec.png'%(logs_path, Origin_Time, epoch+1, i+1))
                    img.save('%s/%s/%s_%simg.png'%(logs_path, Origin_Time, epoch+1, i+1))
                    fig.save('%s/%s/%s_%sfig.png'%(logs_path, Origin_Time, epoch+1, i+1))
            total_loss += loss
            loss_loc += lloc
            loss_conf += lconf
            loss_cls += lcls
            loss_img += limg.item()
            loss_bi += lbi
            waste_time = time.time() - start_time

            pbar.set_postfix(**{'total_loss': total_loss.item() / (iteration + 1),
                                'lr'        : get_lr(optimizer),
                                'step/s'    : waste_time})
            pbar.update(1)

            start_time = time.time()
    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            s_series_val, targets_val, images_val = batch[0], batch[1], batch[2]

            with torch.no_grad():
                if cuda:
                    s_series_val = Variable(s_series_val).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets_val]
                    images_val = Variable(images_val).cuda()
                else:
                    s_series_val = Variable(s_series_val)
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                    images_val = Variable(images_val)
                optimizer.zero_grad()
                outputs = net(images_val)
                losses = []
                lloc,lconf,lcls = 0,0,0
                for i in range(2):
                    loss_item = yolo_losses[i](outputs[i], targets_val)
                    losses.append(loss_item[0])
                    lloc += loss_item[3]
                    lconf += loss_item[1]
                    lcls += loss_item[2]
                reconstruct_val = outputs[2]
                weights_val = outputs[4]
                lbi = 0.01*binary(weights_val)
                limg = 1000*crit(reconstruct_val, images_val)
                losses.append(limg)
                losses.append(lbi)
                loss = sum(losses)
                val_loss += loss
                val_loc += lloc
                val_conf += lconf
                val_cls += lcls
                val_img += limg.item()
                val_bi += lbi
            pbar.set_postfix(**{'total_loss': val_loss.item() / (iteration + 1)})
            pbar.update(1)
    net.train()
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f loc: %.4f conf: %.4f cls: %.4f img: %.4f weights: %.4f\n Val Loss: %.4f loc: %.4f conf: %.4f cls: %.4f img: %.4f weights: %.4f' % (total_loss/(epoch_size),loss_loc/(epoch_size),loss_conf/(epoch_size),loss_cls/(epoch_size),loss_img/(epoch_size),loss_bi/epoch_size,
    val_loss/(epoch_size_val),val_loc/(epoch_size_val),val_conf/(epoch_size_val),val_cls/(epoch_size_val),val_img/(epoch_size_val),val_bi/epoch_size_val))

    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), '%s/%s/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%(logs_path, Origin_Time,(epoch+1),total_loss/(epoch_size),val_loss/(epoch_size_val)))

    with open('%s/%s/output.txt'%(logs_path, Origin_Time), 'a+') as f:
        f.write('Epoch:'+ str(epoch+1) + '/' + str(Epoch) + '\n')
        f.write('Learning rate: %s\n'%(get_lr(optimizer)))
        f.write('Total Loss: %.4f loc: %.4f conf: %.4f cls: %.4f img: %.4f weights: %.4f\n Val Loss: %.4f loc: %.4f conf: %.4f cls: %.4f img: %.4f weights: %.4f' % (total_loss/(epoch_size),loss_loc/(epoch_size),loss_conf/(epoch_size),loss_cls/(epoch_size),loss_img/(epoch_size),loss_bi/epoch_size,
    val_loss/(epoch_size_val),val_loc/(epoch_size_val),val_conf/(epoch_size_val),val_cls/(epoch_size_val),val_img/(epoch_size_val),val_bi/epoch_size_val))

    return (total_loss/(epoch_size),val_loss/(epoch_size_val))

if __name__ == "__main__":
    # The length of the S series
    M = 333
    # Patterns used in the lab 
    image_size_lab = (64,64)

    # trick of the learning rate, not recommanded
    Cosine_lr = False
    # if use gpu to accelerate
    Cuda = True
    smoooth_label = 0 # not necessary
    #-------------------------------------------#
    #   If use pretrained model
    #   Not recommanded for the first time but for interupted
    #-------------------------------------------#
    Use_model = False
    model_path = "logs/Sat_Jul_10_23:21:39_2021/Epoch8-Total_Loss38.3769-Val_Loss37.8774.pth"
    logs_path = 'logs' # where to save the results

    # create the folder for logs
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)
    Origin_Time = time.asctime(time.localtime(time.time())).replace(' ','_')
    if not os.path.exists('%s/%s'%(logs_path,Origin_Time)):
        os.mkdir('%s/%s'%(logs_path,Origin_Time))
    print(Origin_Time)
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)
    if Cuda:
        if torch.cuda.is_available():
            print('Cuda detected.')
        else:
            Cuda = False
            print('Cuda not found, use cpu instead.')
    else:
        print('Attention: You do not use cuda.')
    device = torch.device('cuda' if Cuda else 'cpu')
    #-------------------------------#
    #   Pre-trained yolo info
    #-------------------------------#
    anchors_path = 'model_data/yolo_anchors16.txt'
    classes_path = 'model_data/voc_classes.txt'   
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names)
    
    # Create or copy the model
    model = TrainANet(len(anchors[0]),num_classes, M)
    if Cuda:
        model = model.to(device='cuda')
        #summary(model, (4096,))
    if Use_model:
        print('Loading weights into state dict...')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        keys = []
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if np.shape(model_dict[k]) ==  np.shape(v):
                    keys.append(k)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in keys}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('Finished!')
    else:
        print('No Pretraining model, Using random origin.')

    # init
    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # define loss function
    yolo_losses = []
    for i in range(2):
        yolo_losses.append(YOLOLoss(np.reshape(anchors,[-1,2]),num_classes, 
                                (512,512), smoooth_label, device))
                                # 512 -- the size of anchors
    crit = nn.MSELoss(reduction='mean')
    binary = RestrictionLoss(otherbar=-1)

    # we have tried to freeze some part of the network while training, but it seems not available.
    train_mode = [  #lr,epoch,ifunfreeze
        (1e-2,12,False),
        (1e-2,28,True)
    ]
    Batch_size = 200
    rootpath = 'Fashion_MNIST_train'
    dataset = 'fash'
    
    Epoch = 0
    annotation_path = '%s/train_%s.txt'%(rootpath,dataset)
    # 0.1 for validation, 0.9 for training
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    indexs = np.array(range(len(lines)))
    np.random.seed(10086)
    np.random.shuffle(indexs)
    lines = [lines[idx] for idx in indexs]
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    trainset = [line.split()[0] + '\n' for line in lines[:num_train]]
    valset = [line.split()[0] + '\n' for line in lines[num_train:]]
    with open('trainset.txt','w') as f:
        f.writelines(trainset)
    with open('valset.txt', 'w') as f:
        f.writelines(valset)
    # we use S series in .txt as input
    gen = GenerateInput(Batch_size, lines[:num_train], indexs[:num_train], M, image_size_lab, dataset='fashion').generate()
    gen_val = GenerateInput(Batch_size, lines[num_train:], indexs[num_train:], M, image_size_lab, dataset='fashion').generate()

    for lr,epoch,inlayer in train_mode:
        Begin_Epoch = Epoch
        Epoch += epoch

        for param in model.mainnet.forhead.inlayer.parameters():
            param.requires_grad = inlayer

        optimizer = optim.Adam(net.parameters(),lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=8,gamma=0.1)

        epoch_size = max(1, num_train//Batch_size)
        epoch_size_val = num_val//Batch_size

        
        for epoch in range(Begin_Epoch,Epoch):
            loss, vloss = fit_ont_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,gen_val,Epoch,Cuda)
            lr_scheduler.step()
