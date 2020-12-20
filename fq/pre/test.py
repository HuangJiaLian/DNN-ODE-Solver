# 测试matplotlib

import matplotlib.pyplot as plt 

x = [1,2,3,4,5,6,7,8]
y = [1,4,9,16,25,36,49,64]

plt.title('Hello Matplotlib.')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.plot(x,y,label='$y = x^2$')
plt.legend()
plt.savefig('HelloPlt.png')
plt.savefig('HelloPlt.eps')
plt.show()


# Test numpy
import numpy as np 
x = np.linspace(0,10, 50)
y = np.sin(x)

plt.title('Sin Function')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.plot(x,y,label='$y = \sin (x)$')
plt.legend()
plt.savefig('sin.png')
plt.savefig('sin.eps')
plt.show()


# Test Tensorflow
import tensorflow as tf 
print(tf.__version__)

