import random
with open('fashion0909/test_fash.txt','r') as f:
    w  = f.readlines()

random.seed(6708)
random.shuffle(w)
random.seed(None)

with open('fashion0909/test_fash.txt', 'w') as f:
    f.writelines(w[:100])