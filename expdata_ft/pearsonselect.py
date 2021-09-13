from scipy.stats import pearsonr

with open('expdata_ft/test.txt', 'r') as f:
    testlist = f.read().split('\n')[:-1]
for name in testlist:
    name = name.split(' ')[0]
    with open(name, 'r') as f:
        ten = f.read().split('\n')[:-1]
    with open(name.replace('Object/', 'Object_ref/').replace('Objects', 'Objects_ref'), 'r') as f:
        ref = [int(o) for o in f.read().split('\n')[:-1]]
    pr = []
    for one in ten:
        one = [int(o) for o in one.split(' ')]
        pr.append(pearsonr(one,ref)[0])
    with open(name, 'w') as f:
        f.write(ten[pr.index(max(pr))])    