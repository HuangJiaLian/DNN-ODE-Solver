# coding: utf-8

##########################
# Jack Huang
# jackhuang.wz@gmail.com
# 2021.7.13
##########################

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import os
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 

MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'diffusion_model'
L = 1
NT = 30
NX = 30
# LR = 1e-4
LR = 0.01
LAYERS = [2, 20, 1]
alpha = 1/6.0
a1 = 1
c1 = 1
c2 = 1
LR = 1e-4


def add_layer(inputs, in_size, out_size, activation = None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    # Because the recommend initial value of biases != 0; so add 0.1
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1) 
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation is None:
        outputs = Wx_plus_b
    else:
        outputs = activation(Wx_plus_b)
    return outputs

def forward(x,t):
    global LAYERS
    # Computing layer not include the input layer
    num_layer = len(LAYERS) - 1
    hs = [None]*num_layer
    # Input 
    net_in = tf.concat([x,t],1)
    h1 = add_layer(net_in, LAYERS[0], LAYERS[1], activation = tf.nn.sigmoid)
    out = add_layer(h1, LAYERS[1], LAYERS[2], activation=None)
    return out

xs = tf.placeholder(tf.float32,shape=(None, 1))
ts = tf.placeholder(tf.float32,shape=(None, 1))

Q = forward(xs, ts)
Q_t = tf.gradients(Q, ts)[0]
Q_x = tf.gradients(Q, xs)[0]
Q_xx = tf.gradients(Q_x, xs)[0]

SSEu = a1 * tf.reduce_mean(tf.reduce_sum(tf.square(Q_t - alpha*Q_xx)))

zeros = np.zeros([NT*NX,1])
ones = np.ones([NT*NX,1])
Ls = L * ones 

SSEc1 = c1 * tf.reduce_mean(tf.reduce_sum(tf.square(forward(xs, zeros) - 1)))
SSEc2 = c2 * tf.reduce_mean(tf.reduce_sum(tf.square(forward(xs, ts) - forward(xs+L, ts))))

loss = SSEu + SSEc1 + SSEc2

train_step = tf.train.AdadeltaOptimizer(LR).minimize(loss)

# Trainning Data
arr = []
for x in range(NX):
    for t in range(NT):
        b = [x/(NX-1), t/(NT-1), 2*x/(NX-1)]
        arr.append(b)

data = np.array(arr)
t = data[:,1][:, np.newaxis]
x = data[:,0][:, np.newaxis]
x2 = data[:,2][:, np.newaxis]


saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Model restored")

fig = plt.figure()  
ax = Axes3D(fig)  
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('q')  

T, X = np.meshgrid(np.linspace(0,1,NT),np.linspace(0, 2, NX))
print(T.shape)
print(X.shape)
plt.ion()


for i in range(400000000000):
    sess.run(train_step, feed_dict={xs:x, ts:t})    
    if i%100 == 0:
            print(sess.run(loss, feed_dict={xs:x, ts:t}))
            # 保存训练模型
            if not os.path.exists(MODEL_SAVE_PATH):
                os.makedirs(MODEL_SAVE_PATH)
            saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME))
                    
            Z = sess.run(Q, feed_dict={xs:x2, ts:t})
            Z = Z.reshape(NX,NT,order='C')

            try:
                surfaces.remove()
            except Exception:
                pass
            surfaces = ax.plot_surface(T, X, Z, rstride=1, cstride=1, cmap=plt.cm.jet)
            plt.pause(0.01)