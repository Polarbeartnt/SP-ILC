import os

root = 'Fashion_MNIST_test/label_inv'
filelist = os.listdir(root)

for name in filelist:
    name = root + '/' + name
    with open(name, 'r') as f:
        labels = f.readlines()
    newlabels = []
    for label in labels:
        label = label.split(',')
        newlabel = ' '.join([label[1], label[0], label[3], label[2], label[4]])
        newlabels.append(newlabel)
    with open(name.replace('_inv', ''), 'w') as f:
        f.writelines(newlabels)