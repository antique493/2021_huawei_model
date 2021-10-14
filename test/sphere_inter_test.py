import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt 
from sphere_inter import trilaterate  

def main():
    p1 = np.array([0,0,1300])
    p2 = np.array([5000,0,1700])
    p3 = np.array([0,5000,1700])
    p4 = np.array([5000,5000,1300])
    r1 = 750
    r2 = 4550
    r3 = 4550
    r4 = 6300

    a,b = trilaterate(p1, p2, p3, r1, r2, r3)
    print("1: ", a)
    print("2: ", b)

    a,b = trilaterate(p1, p2, p4, r1, r2, r4)
    print("1: ", a)
    print("2: ", b)

    a,b = trilaterate(p1, p3, p4, r1, r3, r4)
    print("1: ", a)
    print("2: ", b)

    a,b = trilaterate(p2, p3, p4, r2, r3, r4)
    print("1: ", a)
    print("2: ", b)

if __name__ == '__main__':
    main()
    