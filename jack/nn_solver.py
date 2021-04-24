import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

import matplotlib.pyplot as plt
import numpy as np 
import os 

T0 = 5
# Analytical solution
def psy_analytic(t):
    return -15*np.exp(-0.2*t) + 20

nt = 30
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.plot(np.linspace(0,30,nt), psy_analytic(np.linspace(0,30,nt)), label='Target')
plt.xlabel('Time [s]', fontsize=14)
plt.ylabel('Temperature [°C]', fontsize=14)
plt.legend()
plt.ion()
plt.show()

def add_layer(inputs, in_size, out_size, actication_function = None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    # Initial value of biases != 0
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1) 
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if actication_function is None:
        outputs = Wx_plus_b
    else:
        outputs = actication_function(Wx_plus_b)
    return outputs


# create  training data
t_space = np.linspace(0,30,nt)[:, np.newaxis] 
ts = tf.placeholder(tf.float32, [None, 1])  # * rows, 1 col

zero_space = np.zeros([1,1], dtype=np.float32)
zs = tf.placeholder(tf.float32, [None, 1])  # * rows, 1 col

def forward(t):
    l1 = add_layer(t, 1, 10, actication_function = tf.nn.sigmoid)
    net_out = add_layer(l1, 10, 1, actication_function = None)
    return net_out

T = forward(ts)
Tt = tf.gradients(T,ts)[0]
loss = tf.reduce_mean(tf.square(0.2*(20-T) - Tt)) + tf.square(T[0] - T0)

lr = 0.1
train_step = tf.train.AdadeltaOptimizer(lr).minimize(loss)

saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range (400000):
    sess.run(train_step, feed_dict={ts:t_space})
    if i % 100 == 0:
        loss_value = sess.run(loss,feed_dict={ts:t_space})
        print('loss:', loss_value[0])
        res = sess.run(T, feed_dict={ts:t_space})
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        lines = ax.plot(t_space, res, 'r-', lw = 1, label='NN solution')
        plt.legend()
        plt.pause(0.01)