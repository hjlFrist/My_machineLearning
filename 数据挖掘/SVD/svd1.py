# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import  codecs

def readFile(path):
    # 打开文件（注意路径）
    f = open(path)
    # 逐行进行处理
    first_ele = True
    for data in f.readlines():
        ## 去掉每行的换行符，"\n"
        data = data.strip('\n')
        ## 按照 空格进行分割。
        nums = data.split(",")
        # 添加到 matrix 中。
        #print(nums[1:len(nums)])
        if first_ele:
            ### 将字符串转化为整型数据
            nums = [float(x) for x in nums[1:len(nums)]]
            ### 加入到 matrix 中 。
            matrix = np.array(nums)
            first_ele = False
        else:
            nums = [float(x) for x in nums[1:len(nums)]]
            matrix = np.c_[matrix, nums]
    #print(matrix)
    line = dealMatrix_two(matrix)
    f.close()
    return line

def dealMatrix(data):
    ## 一些基本的处理。
    u, sigma, vt = np.linalg.svd(data)
    print(sigma)
    return sigma
    #sig3 = mat([[sigma[0], 0, 0],
                 # [0, sigma[1], 0],
                #[0, 0, sigma[2]]])
    #print(u[:, :3] * sig3 * vt[:3, :])
    #print("transpose the matrix")
    #matrix = data.transpose()
   # print(matrix)
    #print("matrix trace ")
   # print(np.trace(matrix))

def dealMatrix_two(data):
    matrix = data.T
    a1 = matrix.dot(data)
    a2 = np.linalg.eigvals(a1)
    return a2

def ComSim(date):
    for i in range(0,date.columns.size):
        for j in range(i+1,date.columns.size):
            new_list = list(map(lambda x: x[0]-x[1], zip(date[i], date[j])))
            newlist = list(map(abs, new_list))
            s = sum(newlist)/20
            print(s)
            file_new.write(str(i)+","+str(j)+","+str(s)+'\n')
    file_new.close()
if __name__ == '__main__':
    print('start......')
    file_new = open('C:/Users/Administrator/Desktop/4.txt', 'w+')
    f = 'C:/Users/Administrator/Desktop/data/vec_2/'
    date_frame = pd.DataFrame()
    for i in range(0,30):
        file = f+str(i)+'.txt'
        #print(file)
        TeZhen=readFile(file)
        date_frame[i] = TeZhen
    ComSim(date_frame)