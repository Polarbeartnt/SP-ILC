import os
import random
import xml.etree.ElementTree as ET
import re

rootpath = 'data180k'

def main():
    combo = {'list':[0,1,2], }
    for key in combo:
        middata(combo[key])
        writelist(key)

def middata(mode):
    saveBasePath=r"./"
    
    trainval_percent=1
    train_percent=1
    total_xml = []
    for i in mode:
        temp_xml = os.listdir(r'selfmadelabel/%s'%(i))
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

    num=len(total_xml)  
    list=range(num)  
    tv=int(num*trainval_percent)  
    tr=int(tv*train_percent)  
    trainval= random.sample(list,tv)
    train=random.sample(trainval,tr)  
    
    print("train and val size",tv)
    print("traub suze",tr)
    ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
    ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')  
    ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    fval = open(os.path.join(saveBasePath,'val.txt'), 'w')  
    
    for i  in list:  
        name=total_xml[i][:-4]+'\n'  
        if i in trainval:  
            ftrainval.write(name)  
            if i in train:  
                ftrain.write(name)  
            else:  
                fval.write(name)  
        else:  
            ftest.write(name)  
    
    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest .close()

def writelist(cob):
    sets=['train', 'val', 'test']
    classes = ["0","1","2","3","4","5","6","7","8","9"]
    def convert_annotation(mode, image_id, list_file):
        with open('selfmadelabel/%s/%s.xml'%(mode, image_id),'r',encoding='utf-8') as in_file:
            respXML = in_file.read()
        scrubbedXML = re.sub('&.+[0-9]+;', '', respXML)
        root = ET.fromstring(scrubbedXML)

        for obj in root.iter('object'):
            difficult = 0 
            if obj.find('difficult')!=None:
                difficult = obj.find('difficult').text
                
            cls = obj.find('name').text
            if cls not in classes or int(difficult)==1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            x1 = int(xmlbox.find('xmin').text)
            y1 = int(xmlbox.find('ymin').text)
            x2 = int(xmlbox.find('xmax').text)
            y2 = int(xmlbox.find('ymax').text)
            b = (int((x1+x2)/2), int((y1+y2)/2), int(x2-x1), int(y2-y1))
            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

    for image_set in sets:
        image_ids = open('%s.txt'%(image_set)).read().strip().split()
        list_file = open('%s_%s.txt'%(image_set, cob), 'w')
        for image_id in image_ids:
            list_file.write('%s/selfmadetxt/%s/%s.txt'%(rootpath,image_id[0], image_id))
            convert_annotation(image_id[0], image_id, list_file)
            list_file.write('\n')
        list_file.close()

if __name__ == '__main__':
    main()
