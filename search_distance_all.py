from ast import NameConstant
import calculate_postion
import numpy as np
import matplotlib.pyplot as plt
from pso import PSO

input_data = 0
output_data = 0

def fit_fun(X):  # 适应函数
    leas_res, opti_res = calculate_postion.get_result(input_data, output_data-np.array([X]).reshape(-1,1))
    sub_res = ((opti_res[:,0]-input_data)**2).sum(axis=1, keepdims=True)
    return (abs(sub_res - output_data**2)).sum()

def pso_optimize():
    dim = 4
    size = 20
    iter_num = 300
    x_max = 200
    max_vel = 10
    pso = PSO(dim, size, iter_num, x_max, max_vel, fit_fun)
    fit_var_list, best_pos = pso.update()
    print("best_distance: ", fit_var_list[-1])
    return best_pos

def readfile():
    files = []
    for i in range(1,325):
        files.append("test/数据集/正常数据清洗去重/"+str(i)+".正常.txt")
    
    for id in range(len(files)):
        file = files[id]
        res = []
        with open(file, "r") as f:
            lines = f.readlines()[1:]
            for line in lines:
                nums = line.split(":")
                for a in nums:
                    res.append(int(a))

        output_data_all = np.array(res).reshape(-1,1)
        global input_data
        input_data = np.array([[0,0,1300],[5000,0,1700],[0,5000,1700],[5000,5000,1300]])
        result_least = np.zeros((3,0))
        result_optimize = np.zeros((3,0))
        for i in range(int(output_data_all.shape[0]/4)):
            global output_data
            output_data = output_data_all[i*4:i*4+4]
            final_res1 = np.zeros((3,1))
            final_res2 = np.zeros((3,1))
            best_change = pso_optimize()
            final_res1, final_res2 = calculate_postion.get_result(input_data, output_data-np.array([best_change]).reshape(-1,1))

            print("curr id: ", i)
            print("least_res: ", final_res1)
            print("optimize_res: ", final_res2)
            # origin_res1, orgin_res2 = calculate_postion.get_result(input_data, output_data)
            # print("least_res_ori: ", origin_res1)
            # print("optimize_res_ori: ", orgin_res2)
            result_least = np.concatenate((result_least, final_res1), axis=1)
            result_optimize = np.concatenate((result_optimize, final_res2), axis=1)

        # N,3
        # print("final result_least: ", result_least.transpose())
        # print("final result_optimize: ", result_optimize.transpose())

        # save_result
        with open("test/数据集/正常数据结果/"+str(id+1)+".正常.txt", "w") as f:
            for k in range(result_optimize.shape[1]):
                for l in range(result_optimize.shape[0]):
                    res = result_optimize[l,k]
                    f.write(str(res)+' ')
                f.write("\n")
        print("Finish file: ", id+1)


if __name__ == '__main__':
    readfile()
    

    