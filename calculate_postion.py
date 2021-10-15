import numpy as np
from least_square import least_square_res
from lm_optimizer import lm_optimizer

def distance_function(params, input_data, output_data, weight):
    x = params[0,0]
    y = params[1,0]
    z = params[2,0]
    res = (x-input_data[:,0:1])**2+(y-input_data[:,1:2])**2+(z-input_data[:,2:3])**2 - output_data**2
    return np.dot(weight, res)

def get_result(point, distance):
    # distance 是一个numpy 大小为[4,1],代表距离四个点的距离
    # point代表四个点的坐标，大小为[4,3]
    # least_result 是最小二乘结果
    # optimize_result 是优化结果
    least_result = least_square_res(point, distance)
    optimizer = lm_optimizer(distance_function)
    # 协方差矩阵
    covariance = np.diag((1/distance)[:,0])
    # covariance = np.diag([1,2,3]).reshape(1,-1).repeat(output_data, axis=0).reshape(-1,3)
    # covariance = np.eye(output_data.shape[0])
    optimize_result, residual_memory = optimizer.LM(least_result, point, distance, covariance)
    return least_result, optimize_result

def readfile():
    res = []    
    with open("test/数据集/正常数据清洗/1.正常.txt", "r") as f:
        lines = f.readlines()[1:]
        for line in lines:
            nums = line.split(":")
            for a in nums:
                res.append(int(a))

    output_data_all = np.array(res).reshape(-1,1) + 50
    input_data_all = np.array([[0,0,1300],[5000,0,1700],[0,5000,1700],[5000,5000,1300]])
    input_data_all = input_data_all.reshape(1,-1).repeat(output_data_all.shape[0]/4, axis=0).reshape(-1,3)
    result_least = np.zeros((3,0))
    result_optimize = np.zeros((3,0))
    for i in range(int(input_data_all.shape[0]/4)):
        input_data = input_data_all[i*4:i*4+4]
        output_data = output_data_all[i*4:i*4+4]
        result, res_params = get_result(input_data, output_data)
        result_least = np.concatenate((result_least, result), axis=1)
        result_optimize = np.concatenate((result_optimize, res_params), axis=1)

        print("least result: ", result)
        print("optimize result: ", res_params)


        
    print("final result_least: ", result_least.max(axis=1), " ", result_least.min(axis=1))
    print("final result_optimize: ", result_optimize.max(axis=1)," ", result_optimize.min(axis=1))


if __name__ == '__main__':
    readfile()