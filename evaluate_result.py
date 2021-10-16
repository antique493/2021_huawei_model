from ast import NameConstant
import calculate_postion
import numpy as np
import matplotlib.pyplot as plt

input_data = 0
output_data = 0

def read_tags():
    tags = []
    with open("test/数据集/Tag坐标信息.txt", encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            results  = line.split()
            tag = []
            for i in range(1,len(results)):
                a = int(results[i])
                tag.append(a)
            tags.append(tag)
    # [N, 3]
    return np.array(tags)

def readfile():
    error_x_s = []
    error_y_s = []
    error_z_s = []
    error_2_s = []
    error_3_s = []
    files = []
    tags = read_tags()
    for i in range(1,325):
        files.append("test/数据集/正常数据结果/"+str(i)+".正常.txt")
    
    for id in range(len(files)):
        file = files[id]
        # [N,3]
        all_res = []
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                res = []
                nums = line.split()
                for a in nums:
                    res.append(float(a))
                all_res.append(res)
        
        result_np = np.array(all_res)

        tag_gt = tags[id]*10

        error = abs(tag_gt - result_np)

        error_2 = (error[0:2]**2).sum(axis=1)**0.5

        error_3 = (error**2).sum(axis=1)**0.5
        print("File: ", id+1, " with error: ", error[0].mean(), " : ", error[1].mean(), " : ", error[2].mean(), " : ", error_2.mean(), " : ",error_3.mean())

        error_x_s.append(error[0].mean())
        error_y_s.append(error[1].mean())
        error_z_s.append(error[2].mean())
        error_2_s.append(error_2.mean())
        error_3_s.append(error_3.mean())
    
    with open("test/数据集/error_res.txt", "w") as f:
        for id_error in range(len(error_x_s)):
            error_x = error_x_s[id_error]
            error_y = error_y_s[id_error]
            error_z = error_z_s[id_error]
            error_2 = error_2_s[id_error]
            error_3 = error_3_s[id_error]
            f.write("File: "+str(id+1)+" with error: "+str(error_x)+" : "+str(error_y)+" : "+str(error_z)+" : "+str(error_2)+" : "+str(error_3))
            f.write("\n")
        f.write("mean  x_error: "+str(np.array(error_x_s).mean())+" y_error: "+str(np.array(error_y_s).mean())+" z_error: "+str(np.array(error_y_s).mean()))
        f.write(" 2_error: "+str(np.array(error_2_s).mean())+" 3_error: "+str(np.array(error_3_s).mean()))
        f.write("\n")
        f.write("max  x_error: "+str(np.array(error_x_s).max())+" y_error: "+str(np.array(error_y_s).max())+" z_error: "+str(np.array(error_y_s).max()))
        f.write(" 2_error: "+str(np.array(error_2_s).max())+" 3_error: "+str(np.array(error_3_s).max()))
        f.write("\n")
        f.write("min  x_error: "+str(np.array(error_x_s).min())+" y_error: "+str(np.array(error_y_s).min())+" z_error: "+str(np.array(error_y_s).min()))
        f.write(" 2_error: "+str(np.array(error_2_s).min())+" 3_error: "+str(np.array(error_3_s).min()))
        f.write("\n")
        f.write("median  x_error: "+str(np.median(np.array(error_x_s)))+" y_error: "+str(np.median(np.array(error_y_s)))+" z_error: "+str(np.median(np.array(error_y_s))))
        f.write(" 2_error: "+str(np.median(np.array(error_2_s)))+" 3_error: "+str(np.median(np.array(error_3_s))))

if __name__ == '__main__':
    readfile()
    
    