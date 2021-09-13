import os

def mkdir(root, name):
    if not os.path.exists(root):
        os.mkdir(root)
    file = []
    file.append(root + '/' + name)
    file.append(root + '/' + name + '_bbox')
    file.append(root + '/' + name + '_predict')
    for f in file:
        if not os.path.exists(f):
            os.mkdir(f)
