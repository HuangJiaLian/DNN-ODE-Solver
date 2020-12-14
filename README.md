# DNN-ODE-Solver
Use deep neural network(DNN) to obtain numerical solution of an ordinary differential equation(ODE).

## Get general solution of ODE, takes the first-order differential equation as an example.
### dy/dx + P(x)y = Q(x)
### The corresponding homogeneous equation is:
### dy/dx + P(x) = 0,
### and the solution is:
### y = Ce^-∫P(x)dx,
### let C=u(x), we have:
### y = u(x)e^-∫P(x)dx.
### Bringing into the original equation:
### u′(x) = Q(x)/e^∫P(x)dx
### Integrate u’(x) to get u(x) and bring it in to get the general solution form as:
### y = Ce^-∫P(x)dx + e^-∫P(x)dx∫Q(x)e^∫P(x)dxdx
### where C is a constat, determined by the initial conditions of the function.