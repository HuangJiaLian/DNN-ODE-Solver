## Function code

```python
import numpy as np 
import matplotlib.pyplot as plt 
t=np.linspace(0,30)
T=-15*np.exp(-0.2*t)+20
plt.title('Function')
plt.xlabel('t')
plt.ylabel('T')
plt.grid()
plt.plot(t,T,label='$T= e^{-0.2t}+20$')
plt.legend()
plt.savefig('function.png')
plt.savefig('function.eps')
plt.show()
```

![](.\function.png)