from shutil import copyfile

with open('expdata_ft/train_ft.txt', 'r') as f:
    train = f.read().split('\n')[:-1]
newtrain = []
for tr in train:
    newtrain += [tr.replace('.txt', '_%s.txt'%(one)) + '\n' for one in range(10)]
filelist = [f.split(' ')[0] for f in train]
for name in filelist:
    with open(name, 'r') as f:
        ten = f.read()
    for ord in range(10):
        '''with open(name.replace('.txt', '_%s.txt'%(ord)), 'w') as f:
            f.write(ten[ord])'''
        copyfile(name.replace('Objects/', 'Objects_fig/').replace('Object/', 'Object_fig/').replace('.txt', '.png'), name.replace('Objects/', 'Objects_fig/').replace('Object/', 'Object_fig/').replace('.txt', '_%s.png'%(ord)))
        copyfile(name.replace('Objects/', 'Objects_ref/').replace('Object/', 'Object_ref/'), name.replace('Objects/', 'Objects_ref/').replace('Object/', 'Object_ref/').replace('.txt', '_%s.txt'%(ord)))
with open('expdata_ft/train_ft.txt', 'w') as f:
    f.writelines(newtrain)