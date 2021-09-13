# encoding = utf-8
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import numpy as np
import colorsys
import os
import torch
import torch.nn as nn
from PIL import Image,ImageFont, ImageDraw
from torch.autograd import Variable
from utils.utils import non_max_suppression, DecodeBox, yolo_correct_boxes
from utils.dataloader import GenerateInput

class Predictor(object):
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化分类器
    #---------------------------------------------------#
    def __init__(self, _defaults, **kwargs):
        self.__dict__.update(_defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()
    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    
    #---------------------------------------------------#
    #   获得所有的先验框
    #---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self):
        device = torch.device('cpu')
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = self.MyNetBody(len(self.anchors[0]),len(self.class_names),self.series_length).eval()
        

        print('Loading weights into state dict...')
        if self.FineTune != None:
            self.ft = self.FineTune(self.series_length).eval()
            ft_dict = torch.load(self.FT_log, map_location=device)
            self.ft.load_state_dict(ft_dict)
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        
        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
            if self.FineTune != None:
                self.ft = nn.DataParallel(self.ft)
                self.ft = self.ft.cuda()
    
        print('Finished!')

        self.yolo_decodes = []
        self.anchors_mask = [[3,4,5],[1,2,3]]
        for i in range(2):
            #self.yolo_decodes.append(DecodeBox(np.reshape(self.anchors,[-1,2])[self.anchors_mask[i]], len(self.class_names),  (self.model_image_size[1], self.model_image_size[0])))
            self.yolo_decodes.append(DecodeBox(np.reshape(self.anchors,[-1,2])[self.anchors_mask[i]], len(self.class_names),  self.input_to_net_size))


        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    #---------------------------------------------------#
    #   检测测试结果
    #---------------------------------------------------#
    def get_detect_results(self, pathlist, Origin_Time, ifimg=True):
        S_gen = GenerateInput(1, pathlist, 0, self.series_length, self.model_image_size, train=False, finetune=False, ifstop=True,dataset=self.dataset).generate()
        error_list = []
        for n,testdata in enumerate(S_gen):
            s_series, names = testdata[0], testdata[2][0]
            with torch.no_grad():
                if self.cuda:
                    s_series = Variable(s_series.type(torch.FloatTensor)).cuda()
                else:
                    s_series = Variable(s_series.type(torch.FloatTensor))
                if self.FineTune != None:
                    s_series = self.ft(s_series) + s_series
                outputs = self.net(s_series)

            graph = np.array(outputs[2].cpu().detach())
            graph[graph<0] = 0
            graph = np.reshape(graph, graph.shape[1:]) * 255.0 / graph.max()
            graph = Image.fromarray(graph).convert('L')

            graph.save('test_results/%s_predict/%s.png'%(Origin_Time, names))

            graph = graph.convert('RGB')
            output_list = []
            for i in range(2):
                output_list.append(self.yolo_decodes[i](outputs[i]))
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, len(self.class_names),
                  conf_thres=self.confidence,
                  nms_thres=self.iou)
            try:
                batch_detections = batch_detections[0].cpu().numpy()
            except:
                batch_detections = np.empty(shape=(0,6))
                print('Empty prediction for input %s.'%(names))
                error_list.append(names)
            
            
            top_index = batch_detections[:,4]*batch_detections[:,5] > self.confidence
            top_conf = batch_detections[top_index,4]*batch_detections[top_index,5]
            top_label = np.array(batch_detections[top_index,-1],np.int32)
            top_bboxes = np.array(batch_detections[top_index,:4])
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)
            np.savetxt('test_results/%s_bbox/%s_conf.txt'%(Origin_Time, names), top_conf)
            np.savetxt('test_results/%s_bbox/%s.txt'%(Origin_Time, names), np.hstack((top_bboxes,(np.atleast_2d(top_label)).T)), fmt="%i")

            if ifimg:
                image = np.array(graph.resize((512,512),Image.NEAREST))
                # 去掉灰条
                boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),np.array(self.model_image_size))

                font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(6e-2 * np.shape(image)[1] + 0.5).astype('int32'))

                thickness = 3 #(np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0]

                img = Image.fromarray(image)
                for i, c in enumerate(top_label):
                    predicted_class = self.class_names[c]
                    score = top_conf[i]

                    top, left, bottom, right = boxes[i] + [-5,-5,5,5]

                    top = max(0, np.floor(top + 0.5).astype('int32'))
                    left = max(0, np.floor(left + 0.5).astype('int32'))
                    bottom = int(min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32')))
                    right = int(min(np.shape(image)[1], np.floor(right + 0.5).astype('int32')))

                    # 画框框
                    label = '{} {:.2f}'.format(predicted_class, score)

                    draw = ImageDraw.Draw(img)
                    label_size = draw.textsize(label, font)
                    label = label.encode('utf-8')
                    #print(label)
                    
                    if top - label_size[1] >= 0:
                        text_origin = np.array([left, top - label_size[1]])
                    else:
                        text_origin = np.array([left, top + 1])

                    for i in range(thickness):
                        draw.rectangle(
                            [left + i, top + i, right - i, bottom - i],
                            outline=self.colors[self.class_names.index(predicted_class)])
                    draw.rectangle(
                        [tuple(text_origin), tuple(text_origin + label_size)],
                        fill=self.colors[self.class_names.index(predicted_class)])
                    draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
                    del draw
                img.save('test_results/%s/rec_%s.png'%(Origin_Time, names))
                #graph.save('test_results/%s/rec_%s'%(Origin_Time, name.split('/')[-1]))
                
            else:
                with open('test_results/%s/out%s.txt'%(Origin_Time, n+1)) as f:
                    f.write(top_conf)
                    f.write(top_label)
                    f.write(top_bboxes)
            #print('Predict input %s successfully.'%(n+1))


