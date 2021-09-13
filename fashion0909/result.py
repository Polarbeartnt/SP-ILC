import os
from shutil import copyfile

oroot = 'fashion0907/fashion_rec/'
droot = 'fashion0907/fashion_results/'
sroot = 'fashion0907/fig/'
filelist = os.listdir(oroot)

for name in filelist:
    copyfile(oroot + name, droot + name.replace('rec_', '').replace('.png', '_rec.png'))
    copyfile(sroot + name.replace('rec_', ''), droot + name.replace('rec_', ''))