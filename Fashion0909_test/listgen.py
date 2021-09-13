import os

filelist = os.listdir('Fashion0909_test/fig')

lines = []
for name in filelist:
    with open('Fashion0909_test/label/'+name.replace('.png', '.txt'), 'r') as f:
        label = f.read().split('\n')
        lines.append('Fashion0909_test/fig/'+name+' '+' '.join(label)+'\n')

with open('Fashion0909_test/test_fash.txt', 'w') as f:
    f.writelines(lines)

print('Finished.')