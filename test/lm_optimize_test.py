'''
#Implement LM algorithm only using basic python
'''
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt 
from lm_optimizer import lm_optimizer  

#generating the input_data and output_data,whose shape both is (num_data,1)
def generate_data(params, num_data):
    # 产生包含噪声的数据
    x = np.array(np.linspace(0,10,num_data)).reshape(num_data,1)
    mid,sigma = 0,5
    y = function(params,x) + np.random.normal(mid, sigma, num_data).reshape(num_data,1)
    return x,y

def function(params, input_data):
    a = params[0,0]
    b = params[1,0]
    #c = params[2,0]
    #d = params[3,0]
    return a*np.exp(b*input_data)
    #return a*np.sin(b*input_data[:,0])+c*np.cos(d*input_data[:,1])
        
def main():
    #set the true params for generate_data() function
    params = np.zeros((2,1))
    params[0,0]=10.0
    params[1,0]=0.8
    # set the data number
    num_data = 100
    # generate data as requested
    data_input,data_output = generate_data(params,num_data)
    # set the init params for LM algorithm 
    params[0,0]=6.0
    params[1,0]=0.3

    # using LM algorithm estimate params
    
    a = lm_optimizer(function)
    est_params, residual_memory = a.LM(params,data_input, data_output)
    print(est_params)
    a_est=est_params[0,0]
    b_est=est_params[1,0]

    plt.scatter(data_input, data_output, color='b')
    # 生成0-10的共100个数据，然后设置间距为0.1
    x = np.arange(0, 100) * 0.1
    plt.plot(x,a_est*np.exp(b_est*x),'r',lw=1.0)
    plt.xlabel("2018.06.13")
    plt.show()
    
    plt.plot(residual_memory)
    plt.xlabel("2018.06.13")
    plt.show()

if __name__ == '__main__':
    main()


