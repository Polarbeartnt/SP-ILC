import numpy as np
from scipy.stats import pearsonr
import os
import pandas as pd

root = 'expdata'
obj = 'Multiple_Objects'

filelist = os.listdir(root+'/'+obj)
reflist = os.listdir(root+'/'+obj+'_ref')

corr = []
for f in filelist:
    exp = np.loadtxt(root+'/'+obj+'/'+f)
    ref = np.loadtxt(root+'/'+obj+'_ref/'+f)
    thiscor = [pearsonr(ref, ex)[0] for ex in exp]
    thiscor.append(thiscor.index(max(thiscor)))
    corr.append(thiscor)

name_attribute = [str(i) for i in range(10)]+['max']
writerCSV=pd.DataFrame(columns=name_attribute,data=corr)
writerCSV.to_csv('expdata/%s_select.csv'%(obj),encoding='utf-8')