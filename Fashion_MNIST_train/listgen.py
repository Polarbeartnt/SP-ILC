import os

filelist = os.listdir('Fashion_MNIST_train/fig')

lines = []
for name in filelist:
    with open('Fashion_MNIST_train/label/'+name.replace('.png', '.txt'), 'r') as f:
        label = f.read().split('\n')
        lines.append('Fashion_MNIST_train/fig/'+name+' '+' '.join(label)+'\n')

with open('Fashion_MNIST_train/train_fash.txt', 'w') as f:
    f.writelines(lines)

print('Finished.')