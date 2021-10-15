import matplotlib.pyplot as plt 
import numpy as np

a4 = np.array([0,0,1300])
a2 = np.array([5000,0,1700])
a3 = np.array([0,5000,1700])
a1 = np.array([5000,5000,1300])

r1 = 6300
r2 = 4500
r3 = 4500


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

t = np.linspace(0, np.pi * 2, 100)
s = np.linspace(0, np.pi, 100)

t, s = np.meshgrid(t, s)
x1 = np.cos(t) * np.sin(s)*r1+a1[0]
y1 = np.sin(t) * np.sin(s)*r1+a1[1]
z1 = np.cos(s)*r1+a1[2]
ax.plot_surface(x1, y1, z1, rstride=1, cstride=1)
x2 = np.cos(t) * np.sin(s)*r2+a2[0]
y2 = np.sin(t) * np.sin(s)*r2+a2[1]
z2 = np.cos(s)*r2+a2[2]
ax.plot_surface(x2, y2, z2, rstride=1, cstride=1)
x3 = np.cos(t) * np.sin(s)*r3+a3[0]
y3 = np.sin(t) * np.sin(s)*r3+a3[1]
z3 = np.cos(s)*r3+a3[2]
ax = plt.subplot(111, projection='3d')
ax.plot_surface(x3, y3, z3, rstride=1, cstride=1)
plt.show()