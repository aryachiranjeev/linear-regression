
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
x_data=np.random.randn(100)
w_real=0.5
b_real=0.1
noise=np.random.randn(100)*0.2
y_data=np.multiply(w_real,x_data)+b_real+noise
plt.scatter(x_data,y_data,c="g")
plt.show()
num_steps=1000
g=tf.Graph()
with g.as_default():
  x=tf.placeholder(dtype="float32")
  y_true=tf.placeholder(dtype="float32")
  
  #prediction 
  
  w=tf.Variable(np.random.randn(),name="weight")
  b=tf.Variable(np.random.randn(),name="bias")
  y_pred = np.multiply(w,x)+b
  
  #loss
  
  loss=tf.reduce_mean(tf.square(y_true-y_pred))
  
  #gradient 
  learning_rate=0.5
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  train = optimizer.minimize(loss)
  
  init=tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init)
    for steps in range(num_steps):
      sess.run(train,{x:x_data , y_true:y_data})
      if(steps%50==0):
        c=sess.run(loss,{y_true:y_data, x:x_data})
        print("steps:{}, loss:{}, w:{}, b:{}".format(steps,c,sess.run(w),sess.run(b)))
        
    training_loss=sess.run(loss,{ x:x_data,y_true:y_data})
    weight=sess.run(w)
    bias=sess.run(b)
    
    #output
    '''
    steps:0, loss:0.056710176169872284, w:0.5412529110908508, b:0.06996870040893555
steps:50, loss:0.05393878370523453, w:0.5115106105804443, b:0.1086776852607727
steps:100, loss:0.05393878370523453, w:0.5115106105804443, b:0.1086776852607727
steps:150, loss:0.05393878370523453, w:0.5115106105804443, b:0.1086776852607727
steps:200, loss:0.05393878370523453, w:0.5115106105804443, b:0.1086776852607727
steps:250, loss:0.05393878370523453, w:0.5115106105804443, b:0.1086776852607727
steps:300, loss:0.05393878370523453, w:0.5115106105804443, b:0.1086776852607727
steps:350, loss:0.05393878370523453, w:0.5115106105804443, b:0.1086776852607727
steps:400, loss:0.05393878370523453, w:0.5115106105804443, b:0.1086776852607727
steps:450, loss:0.05393878370523453, w:0.5115106105804443, b:0.1086776852607727
steps:500, loss:0.05393878370523453, w:0.5115106105804443, b:0.1086776852607727
steps:550, loss:0.05393878370523453, w:0.5115106105804443, b:0.1086776852607727
steps:600, loss:0.05393878370523453, w:0.5115106105804443, b:0.1086776852607727
steps:650, loss:0.05393878370523453, w:0.5115106105804443, b:0.1086776852607727
steps:700, loss:0.05393878370523453, w:0.5115106105804443, b:0.1086776852607727
steps:750, loss:0.05393878370523453, w:0.5115106105804443, b:0.1086776852607727
steps:800, loss:0.05393878370523453, w:0.5115106105804443, b:0.1086776852607727
steps:850, loss:0.05393878370523453, w:0.5115106105804443, b:0.1086776852607727
steps:900, loss:0.05393878370523453, w:0.5115106105804443, b:0.1086776852607727
steps:950, loss:0.05393878370523453, w:0.5115106105804443, b:0.1086776852607727
 '''
predicted=weight*x_data+bias
print("loss=",training_loss,"weight=",weight,"bias=",bias)
#output
'''
loss= 0.053938784 weight= 0.5115106 bias= 0.108677685
'''
#plot of linearregression graph

plt.scatter(x_data,y_data,c='g',label="dataset")
plt.plot(x_data,predicted,c='r',label="fitted line")
plt.title("linear regression graph")
plt.legend()
plt.show()

