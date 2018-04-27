import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

"""
(n,k)=(parm1,parm2),n,k refer to paper
"""
parm1=2
parm2=2
#one-hot coding feature dim
M = 2**parm2
k = np.log2(M)
k = int(k)
#compressed feature dim
n_channel =parm1
R = k/n_channel
CHANNEL_SIZE = M
train_num=8000

xs = tf.placeholder(tf.float32,[None,CHANNEL_SIZE])
ys = tf.placeholder(tf.float32,[None,CHANNEL_SIZE])

def add_layer(inputs,in_size,out_size,activation_fuction=None):
    weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs,weights) + biases
    if activation_fuction is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_fuction(Wx_plus_b)
    return outputs

def encoder(input_signal):
    h1 = add_layer(input_signal,CHANNEL_SIZE,CHANNEL_SIZE,activation_fuction=tf.nn.relu)
    h2 = add_layer(h1,CHANNEL_SIZE,n_channel,activation_fuction=None)
    return h2


def decoder(channel_signal):
    h3 = add_layer(channel_signal,n_channel,CHANNEL_SIZE,activation_fuction=tf.nn.relu)
    h4 = add_layer(h3,CHANNEL_SIZE,CHANNEL_SIZE,activation_fuction=None)
    return h4

def AWGN(channel_signal):
    # Normalization
    channel_signal = (CHANNEL_SIZE ** 0.5) * (channel_signal / tf.norm(channel_signal,axis=1)[:, None])
    # 7dBW to SNR.
    training_signal_noise_ratio = 5.01187
    # bit / channel_use
    communication_rate = R
    # Simulated Gaussian noise.
    noise = (tf.random_normal(tf.shape(channel_signal)) / ((2 * communication_rate * training_signal_noise_ratio) ** 0.5))
    channel_signal += noise
    return channel_signal

train_labels = np.random.randint(0,CHANNEL_SIZE,train_num)
trian_size = tf.size(train_labels) # get size of labels
labels = tf.expand_dims(train_labels, 1) #expand one dim
indices = tf.expand_dims(tf.range(0, trian_size,1), 1) #generate index
concated = tf.concat([indices, labels] , 1) #concat
train_data = tf.sparse_to_dense(concated, tf.stack([trian_size, CHANNEL_SIZE]), 1.0, 0.0) # one-hot
sess = tf.Session()
train_data = train_data.eval(session=sess)
sess.close()

channel_signal = encoder(xs)
signal_noise = AWGN(channel_signal)
output_signal = decoder(signal_noise)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=output_signal))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


for i in range(20000):
    sess.run(train_step,feed_dict={xs:train_data,ys:train_data})
#tshow the learned representations x
test_data = np.eye(M)
x = (sess.run(channel_signal,feed_dict={xs:test_data}))
x = (n_channel**0.5) * (x / LA.norm(x,axis=1)[:, None])
plt.scatter(x[:,0],x[:,1])
plt.axis((-2.5,2.5,-2.5,2.5))
plt.show()