import numpy as np
import matplotlib.pyplot as plt 

class lm_optimizer(object):
    def __init__(self, function):
        # 最大迭代代数
        self.num_iter_ = 100    
        # 设置优化阈值
        self.tao_ = 10**-3
        self.threshold_stop_ = 10**-15
        self.threshold_step_ = 10**-15
        self.threshold_residual_ = 10**-15
        self.function_ = function
    
    def set_function(self, function):
        # 设置优化函数
        # function 应该输入参数为(param, input_data) 返回 output_data
        # param 代表需要被优化的参数
        # input_data 代表输入的值
        # output_data 代表输出的预测值
        self.function_ = function
    
    #calculating the derive of pointed parameter,whose shape is (num_data,1)
    # TODO: 这部分是可以进行改进的，最好可以写成公式偏导数
    # 计算数值偏导数
    def cal_deriv(self, params, input_data, param_index):
        params1 = params.copy()
        params2 = params.copy()
        params1[param_index,0] += 0.000001
        params2[param_index,0] -= 0.000001
        data_est_output1 = self.function_(params1,input_data)
        data_est_output2 = self.function_(params2,input_data)
        return (data_est_output1 - data_est_output2) / 0.000002

    #calculating jacobian matrix,whose shape is (num_data,num_params)
    def cal_Jacobian(self, params, input_data):
        num_params = np.shape(params)[0]
        num_data = np.shape(input_data)[0]
        J = np.zeros((num_data,num_params))
        for i in range(0,num_params):
                J[:,i] = list(self.cal_deriv(params,input_data,i))
        return J

    #calculating residual, whose shape is (num_data,1)
    def cal_residual(self, params, input_data, output_data):
        data_est_output = self.function_(params,input_data)
        residual = output_data - data_est_output
        return residual

    #get the init u, using equation u=tao*max(Aii)
    def get_init_u(self, A,tao):
        m = np.shape(A)[0]
        Aii = []
        for i in range(0,m):
            Aii.append(A[i,i])
        u = tao*max(Aii)
        return u

    #LM algorithm
    def LM(self, params, input_data, output_data):
        # the number of params
        num_params = np.shape(params)[0]
        # set the init iter count is 0
        k = 0
        #calculating the init residual
        residual = self.cal_residual(params, input_data, output_data)
        #calculating the init Jocobian matrix
        Jacobian = self.cal_Jacobian(params, input_data)
        
        A = Jacobian.T.dot(Jacobian)#calculating the init A
        g = Jacobian.T.dot(residual)#calculating the init gradient g
        stop = (np.linalg.norm(g, ord=np.inf) <= self.threshold_stop_)#set the init stop
        #set the init u
        u = self.get_init_u(A,self.tao_)
        #set the init v=2
        v = 2
        
        # 用于存储所有的res值
        residual_memory = []
        while((not stop) and (k < self.num_iter_)):
            k+=1
            while(1):
                # 计算LM中的hessian矩阵
                Hessian_LM = A + u*np.eye(num_params)
                # calculating the update step
                step = np.linalg.inv(Hessian_LM).dot(g)
                if(np.linalg.norm(step) <= self.threshold_step_):
                    stop = True
                else:
                    #update params using step
                    new_params = params + step
                    #get new residual using new params
                    new_residual = self.cal_residual(new_params,input_data,output_data)
                    rou = (np.linalg.norm(residual)**2 - np.linalg.norm(new_residual)**2) / (step.T.dot(u*step+g))
                    if rou > 0:
                        params = new_params
                        residual = new_residual
                        residual_memory.append(np.linalg.norm(residual)**2)
                        #print (np.linalg.norm(new_residual)**2)
                        #recalculating Jacobian matrix with new params
                        Jacobian = self.cal_Jacobian(params,input_data)
                        #recalculating A
                        A = Jacobian.T.dot(Jacobian)
                        #recalculating gradient g
                        g = Jacobian.T.dot(residual)
                        stop = (np.linalg.norm(g, ord=np.inf) <= self.threshold_stop_) or (np.linalg.norm(residual)**2 <= self.threshold_residual_)
                        u = u*max(1/3,1-(2*rou-1)**3)
                        v = 2
                    else:
                        u = u*v
                        v = 2*v
                if(rou > 0 or stop):
                    break;
        print("stop iter: ", k)
            
            
        return params, residual_memory