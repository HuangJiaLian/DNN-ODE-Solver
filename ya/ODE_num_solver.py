import math
import matplotlib.pyplot as plt

# dT = 0.2*(20 - T)
# T0 = 5

num_solution_1 = []
num_solution_2 = []
num_solution_3 = []
for i in range(30):
    if i == 0:
        T1 = 5
        t = T1
        num_solution_1.append(T1)
    else:
        dT1 = 0.2*(20 - t)
        T1 = T1 + i*dT1
        t = T1
        num_solution_1.append(T1)

x2 = []
for i in range(290):
    if i == 0:
        T2 = 5
        t = T2
        num_solution_2.append(T2)
        x2.append(0.1*i)
    else:
        dT2 = 0.2*(20 - t)
        T2 = T2 + 0.1*i*dT2
        t = T2
        num_solution_2.append(T2)
        x2.append(0.1*i)


# T3 = -15*math.exp(-0.2*t) + 20
for t in range(30):
    T3 = -15*math.exp(-0.2*t) + 20
    num_solution_3.append(T3)

plt.grid()
plt.plot(range(30),num_solution_1, label='$\delta t=1$')
plt.plot(x2,num_solution_2, label='$\delta t=0.1$')
plt.plot(range(30),num_solution_3, label='analytical solution')
plt.xlabel('t', fontsize=14)
plt.ylabel('T', fontsize=14)
plt.legend()
plt.show()