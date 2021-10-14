import numpy as np
from least_square import least_square_res
from lm_optimizer import lm_optimizer

def distance_function(params, input_data, weight):
    x = params[0,0]
    y = params[1,0]
    z = params[1,0]
    return np.dot(weight, (x-input_data[:,0:1])**2+(y-input_data[:,1:2])**2+(z-input_data[:,2:3])**2)

def readfile():
    res = []
    with open("test/正常数据/200.正常.txt", "r") as f:
        lines = f.readlines()[1:]
        for line in lines:
            a = line.split(":")
            res.append([int(a[5])])

    output_data = np.array(res)
    input_data = np.array([[0,0,1300],[5000,0,1700],[0,5000,1700],[5000,5000,1300]])
    input_data = input_data.reshape(1,-1).repeat(output_data.shape[0]/4, axis=0).reshape(-1,3)
    result = least_square_res(input_data, output_data)
    print("least result: ", result)

    optimizer = lm_optimizer(distance_function)
    # 协方差矩阵
    covariance = np.eye(output_data.shape[0])
    res_params, residual_memory = optimizer.LM(result, input_data, output_data**2, covariance)
    print("optimize result: ", res_params)


if __name__ == '__main__':
    readfile()