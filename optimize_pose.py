import numpy as np
from least_square import least_square_res
from lm_optimizer import lm_optimizer

def distance_function(params, input_data, output_data, weight):
    x = params[0,0]
    y = params[1,0]
    z = params[2,0]
    k1 = params[3,0]
    b1 = params[4,0]
    k2 = params[5,0]
    b2 = params[6,0]
    d_new = (output_data[0:input_data.shape[0]]-params[7:7+input_data.shape[0],0:])/params[7+input_data.shape[0]:,0:]
    res = (((x-input_data[:,0:1])**2+(y-input_data[:,1:2])**2+(z-input_data[:,2:3])**2)**0.5 - d_new)**2
    res_1 = 1/(k1 * output_data[0:input_data.shape[0]] + b1) * (params[7:input_data.shape[0]+7])
    res_2 = 1/(k2 * output_data[0:input_data.shape[0]] + b2) * (params[7+input_data.shape[0]:]-1)
    res_final = np.concatenate((res, res_1, res_2), axis=0)

    return np.dot(weight, res_final)

def readfile():
    res = []
    with open("test/数据集/data/正常数据清洗/117.正常.txt", "r") as f:
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

    # x, y, z, k1, b1, k2, b2, sigma1(shape0), sigma2(shape0)
    params_init = np.zeros((3+output_data.shape[0]*2+4,1))
    params_init[0:3,0:] = result
    params_init[3,0] = 1/output_data.mean()
    params_init[4,0] = 0
    params_init[5,0] = 1/output_data.mean()
    params_init[6,0] = 0
    params_init[7:output_data.shape[0]+7,0] = 0
    params_init[7+output_data.shape[0]:,0] = 1

    output_data_new = np.concatenate((output_data, np.zeros((2*output_data.shape[0],1))), axis=0)
    # output_data_new = np.zeros((3*output_data.shape[0],1))
    # 协方差矩阵
    # covariance = np.diag([1,2,3]).reshape(1,-1).repeat(output_data, axis=0).reshape(-1,3)
    covariance = np.eye(output_data_new.shape[0])
    covariance[0:output_data.shape[0]] *= 1
    res_params, residual_memory = optimizer.LM(params_init, input_data, output_data_new, covariance)
    print("optimize result: ", res_params)


if __name__ == '__main__':
    readfile()