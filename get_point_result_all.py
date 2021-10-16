from ast import NameConstant
import calculate_postion
import numpy as np
import matplotlib.pyplot as plt

input_data = 0
output_data = 0

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
            # output_data[0,0] -= 0.03037*output_data[0,0] - 146  
            # output_data[1,0] -= 0.02067*output_data[1,0] - 115.8  
            # output_data[2,0] -= 0.02342*output_data[2,0] - 138.6  
            # output_data[3,0] -= 0.02822*output_data[3,0] - 161.2  
            final_res1 = np.zeros((3,1))
            final_res2 = np.zeros((3,1))

            final_res1, final_res2 = calculate_postion.get_result(input_data, output_data)

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
    
    