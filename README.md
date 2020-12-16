# DNN-ODE-Solver
Use deep neural network(DNN) to obtain numerical solution of an ordinary differential equation(ODE).

## Get general solution of ODE
Let's take the first-order differential equation as an example. 

$$
\frac{dy}{dx} + P(x)y = Q(x)
$$

The corresponding homogeneous equation is

$$
\frac{dy}{dx} + P(x)y = 0
$$

make some  transposition:

$$
\frac{dy}{y} = P(x)dx
$$

Integral over the equation and the solution is
$$
y = Ce^{-\int P(x)dx}
$$

The inhomogeneous equation is:
$$
\frac{dy}{dx} + P(x)y = Q(x)
$$

let $C=u(x)$, we have
$$
y = u(x)e^{-\int P(x)dx}
$$
$$
\frac{dy}{dx} = u′(x)e^{-∫P(x)dx}-u(x)e^{-∫P(x)dx} * Q(x)
$$


Bringing into the original equation
$$
u′(x) = \frac{Q(x)}{e^{∫P(x)dx }}
$$

Integrate $u’(x)$ to get $u(x)$ and bring it in to get the general solution form as: 
$$
y = Ce^{-∫P(x)dx} + e^{-∫P(x)dx}∫Q(x)e^{∫P(x)dx}dx
$$
where $C$ is a constat, determined by the initial conditions of the function.  

Ref:

[Solution of First Order Linear Differential Equations](https://www.mathsisfun.com/calculus/differential-equations-first-order-linear.html)
