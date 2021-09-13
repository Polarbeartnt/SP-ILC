import os

filelist = os.listdir('fashion0909/fig')

lines = []
for name in filelist:
    with open('fashion0909/label/'+name.replace('.png', '.txt'), 'r') as f:
        label = f.read().split('\n')
        lines.append('fashion0909/fig/'+name+' '+' '.join(label)+'\n')

with open('fashion0909/train_fash.txt', 'w') as f:
    f.writelines(lines)

print('Finished.')