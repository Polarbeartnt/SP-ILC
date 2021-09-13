import os

root = 'dataset/'
Path = ['double_mnist', 'triple_mnist']

for path in Path:
    files = ['DouTriMnist/'+root+path+'/test/'+n+'\n' for n in os.listdir(root+path+'/test')]
    with open('test_%s.txt'%(path), 'w') as f:
        f.writelines(files)