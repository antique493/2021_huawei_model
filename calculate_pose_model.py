import numpy as np
from least_square import least_square_res
from lm_optimizer import lm_optimizer

def distance_function(params, input_data, output_data, weight):
    k1 = 10000
    k2 = 10000
    b1 = 0
    b2 = 0
    m = 0
    x = params[0,0]
    y = params[1,0]
    z = params[2,0]
    sig1 = params[3:3+output_data.shape[0]//3,0:]
    sig2 = params[3+output_data.shape[0]//3:3+2*output_data.shape[0]//3,0:]
    res = (x-input_data[:,0:1])**2+(y-input_data[:,1:2])**2+(z-input_data[:,2:3])**2 - ((output_data[0:output_data.shape[0]//3]-m-sig1)/sig2)**2
    res_1 = (sig1)*(1/(k1*output_data[0:output_data.shape[0]//3]+b1))**0.5
    res_2 = (sig2-1)*(1/(k2*output_data[0:output_data.shape[0]//3]+b2))**0.5
    res_final = np.concatenate((res, res_1, res_2), axis=0)
    return np.dot(weight, res_final)

def readfile():
    res = []
    with open("test/数据集/data/正常数据清洗/172.正常.txt", "r") as f:
        lines = f.readlines()[1:]
        for line in lines:
            a = line.split(":")
            res.append([int(a[5])])

    output_data_all = np.array(res)
    input_data_all = np.array([[0,0,1300],[5000,0,1700],[0,5000,1700],[5000,5000,1300]])
    input_data_all = input_data_all.reshape(1,-1).repeat(output_data_all.shape[0]/4, axis=0).reshape(-1,3)
    result_least = np.zeros((3,0))
    result_optimize = np.zeros((3,0))
    for i in range(int(input_data_all.shape[0]/4)):
        input_data = input_data_all[i*4:i*4+4]
        output_data = output_data_all[i*4:i*4+4]
        result = least_square_res(input_data, output_data)
        print("least result: ", result)
        result_least = np.concatenate((result_least, result), axis=1)

        optimizer = lm_optimizer(distance_function)
        # 协方差矩阵
        # covariance = np.diag((1/output_data)[:,0])
        param_optimize = np.concatenate((result, np.zeros((output_data.shape[0],1)), np.ones((output_data.shape[0],1)) ), axis=0)
        
        output_data_new = np.concatenate((output_data, np.zeros((2*output_data.shape[0],1))), axis=0)
        # covariance = np.diag([1,2,3]).reshape(1,-1).repeat(output_data, axis=0).reshape(-1,3)
        covariance = np.eye(output_data_new.shape[0])
        covariance[0:output_data_new.shape[0]//3,0:output_data_new.shape[0]//3] = np.diag((1/output_data**2)[:,0])
        res_params, residual_memory = optimizer.LM(param_optimize, input_data, output_data_new, covariance)
        print("optimize result: ", res_params)
        result_optimize = np.concatenate((result_optimize, res_params[0:3,:]), axis=1)
    print("final result_least: ", result_least.max(axis=1), " ", result_least.min(axis=1))
    print("final result_optimize: ", result_optimize.max(axis=1)," ", result_optimize.min(axis=1))


if __name__ == '__main__':
    readfile()