import numpy as np

def Hadmard(rank): # 生成Hadmard矩阵
    def sig(i, j):
        temp = i & j
        result = 0
        for step in range(4):
            result += ((temp >> step) & 1)
        if 0 == result % 2:
            sign = 1
        else:
            sign = -1
        return sign
    
    generate = np.ones((rank, rank), dtype = np.float32)
    for i in range(rank):
        for j in range(rank):
            generate[i][j] = sig(i, j)
    return generate

def GenerateSample(): # 随机生成样例物品
    img = np.random.randint(0,2,(4,4))
    print('生成图片:\n', img)
    return img

if __name__ == '__main__': # 模拟生成图片，作用A矩阵后，复原图片
    Figure = GenerateSample()
    A = Hadmard(16)
    S = np.dot(A, np.reshape(Figure, [16,]))
    print('S序列', S)
    O = np.dot(np.linalg.inv(A), S)
    print('还原图片：\n', np.reshape(O, [4,4]))