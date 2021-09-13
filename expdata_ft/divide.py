import random

filesingle = ['expdata_ft/Single_Object/%03d.txt'%(n) for n in range(139)]
filemultiple = ['expdata_ft/Multiple_Objects/%03d.txt'%(n+139) for n in range(139)]

ex3000 = filemultiple[:79]
ex4000 = filemultiple[79:]

random.seed(708)
random.shuffle(filesingle)
random.shuffle(ex3000)
trainsets = filesingle[40:] + ex3000[40:] + ex4000
random.seed(6708)
random.shuffle(trainsets)
random.seed(None)
testsets = filesingle[:40] + ex3000[:40]

with open('expdata_ft/train_ft.txt', 'w') as f:
    for line in trainsets:
        if line.find('jects') != -1:
            boxname = line.replace('jects', 'jects_label')
        else:
            boxname = line.replace('ject', 'ject_label')
        with open(boxname, 'r') as g:
            box = g.readlines()
        for i in range(len(box)):
            b = box[i].replace('\n','').split(' ')
            box[i] = [int(l) for l in b]
            b[0] = (box[i][0] + box[i][2])/2
            b[1] = (box[i][1] + box[i][3])/2
            b[2] = box[i][2] - box[i][0]
            b[3] = box[i][3] - box[i][1]
            b = [str(int(l//8)) for l in b[:4]] + [str(box[i][4])]
            box[i] = ','.join(b)
        box = ' '.join(box)
        f.write(line + ' ' + box + '\n')

with open('expdata_ft/test.txt', 'w') as f:
    for line in testsets:
        if line.find('jects') != -1:
            boxname = line.replace('jects', 'jects_label')
        else:
            boxname = line.replace('ject', 'ject_label')
        with open(boxname, 'r') as g:
            box = g.readlines()
        for i in range(len(box)):
            b = box[i].replace('\n','').split(' ')
            box[i] = [int(l) for l in b]
            b[0] = (box[i][0] + box[i][2])/2
            b[1] = (box[i][1] + box[i][3])/2
            b[2] = box[i][2] - box[i][0]
            b[3] = box[i][3] - box[i][1]
            b = [str(int(l//8)) for l in b[:4]] + [str(box[i][4])]
            box[i] = ','.join(b)
        box = ' '.join(box)
        f.write(line + ' ' + box + '\n')