import util 
import matplotlib.pyplot as plt 
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D

NX = 33
NT = 33
c = 7.0

u = util.get_num_solution(NX,NT,7.)
u = u.reshape(NX,NT,order='C')

'''
fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('T')
ax.set_ylabel('X')
ax.set_zlabel('U')
T,X = np.meshgrid(np.linspace(0,1,NT),np.linspace(0,1,NX))
surfaces = ax.plot_surface(T,X,u,rstride=1,cstride=1,cmap=plt.cm.jet)
'''

x= np.linspace(0,1,NX)
t= np.linspace(0,1,NT)

fig_c = plt.figure()
ax_c = fig_c.add_subplot(1,1,1)
for i in range(0,32,5):
    ax_c.scatter(x,u[:,i], s=15, alpha = 0.1,marker='o',label='t='+str(i))
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.show()