import os

filelist = os.listdir('Fashion_MNIST_test/fig')

lines = []
for name in filelist:
    with open('Fashion_MNIST_test/label/'+name.replace('.png', '.txt'), 'r') as f:
        label = f.read().split('\n')
        lines.append('Fashion_MNIST_test/fig/'+name+' '+' '.join(label)+'\n')

with open('Fashion_MNIST_test/test_fash.txt', 'w') as f:
    f.writelines(lines)

print('Finished.')