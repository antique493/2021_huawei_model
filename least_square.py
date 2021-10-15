import numpy as np

def least_square_res(input_data, output_data, max_iter=100):
    # A*X=L
    # factor 参数
    c = 10
    iter = 0
    P = np.eye(input_data.shape[0]-1)
    result = np.zeros(3)
    while (iter < max_iter):
        # section 2.1 In <An Indoor Localization Method for Pedestrians Base on Combined UWB/PDR/Floor Map> 
        A = input_data[1:] - input_data[0:-1]
        L = ((input_data[1:]**2).sum(axis=1, keepdims=True) - (input_data[0:-1]**2).sum(axis=1, keepdims=True) \
            + output_data[0:-1]**2 - output_data[1:]**2) * 0.5
        
        # result is 3*1
        # (A^T P A)^-1 A^T P L
        result = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(A.transpose(), P), A)+0.5), A.transpose()), P), L)
        resdiual = abs(np.dot(A, result) - L)
        # print("resdiual: ", resdiual.mean())
        # [X,1]
        if np.median(resdiual) == 0:
            U = resdiual/c/np.mean(resdiual)
        else:
            U = resdiual/c/np.median(resdiual)
        W_U = U.copy()
        W_U[abs(U)<=1] = ((1-U**2)**2)[abs(U)<=1]
        W_U[abs(U)>1] = 0
        P = np.diag(W_U[:,0])
        # print(P.mean())
        iter += 1
    return result



    
