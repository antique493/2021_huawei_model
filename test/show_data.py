import numpy as np
import glob
import matplotlib.pyplot as plt 

def read_tags():
    tags = []
    with open("./数据集/Tag坐标信息.txt", encoding='UTF-8') as f:
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

def read_files():
    # estimate_results [files, 4, N]
    estimate_results = []
    files = []
    for i in range(1,325):
        files.append("数据集/正常数据清洗/"+str(i)+".正常.txt")
    for file in files:
        with open(file,"r") as f:
            lines = f.readlines()[1:]
            file_results = [[],[],[],[]]
            for line in lines:
                res = line.split(":")
                for i in range(len(res)):
                    file_results[i].append(int(res[i]))
            estimate_results.append(file_results)
            # print(file)
            # print(len(file_results[0]))
            # print(len(file_results[1]))
            # print(len(file_results[2]))
            # print(len(file_results[3]))

    # [file, 4, N]
    return estimate_results


if __name__ == '__main__':
    input_data = np.array([[0,0,1300],[5000,0,1700],[0,5000,1700],[5000,5000,1300]])
    tags = read_tags()
    distances = read_files()

    for which in range(4):
        x = []
        y = []
        for i in range(tags.shape[0]):
            tag = tags[i]
            distance_gt = ((tag - input_data/10)**2).sum(axis=1)**0.5*10
            
            pose = distances[i]
            x += pose[which]
            y += list((pose[which]-distance_gt[which]))
        
        plt.scatter(x, y)
        plt.show()

    