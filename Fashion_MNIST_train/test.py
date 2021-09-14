import random
with open('Fashion_MNIST_train/test_fash.txt','r') as f:
    w  = f.readlines()

random.seed(6708)
random.shuffle(w)
random.seed(None)

with open('Fashion_MNIST_train/test_fash.txt', 'w') as f:
    f.writelines(w[:100])