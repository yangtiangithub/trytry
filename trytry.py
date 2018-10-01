import tensorflow as tf

# with tf.Graph().as_default(), tf.Session() as sess:
#   matrix1 = tf.Variable(tf.random_uniform([3, 1], maxval=7, minval=1, dtype=tf.int32))
#   matrix2 = tf.Variable(tf.random_uniform([3, 1], maxval=7, minval=1, dtype=tf.int32))
#   matrix3 = tf.add(matrix1, matrix2)
#   matrix_result = tf.concat([matrix1, matrix2, matrix3], axis = 1)
#
#   sess.run(tf.global_variables_initializer())
#   print(matrix1.eval())
#   print(matrix2.eval())
#   print(matrix3.eval())
#   print(matrix_result.eval())

# with tf.Graph().as_default(), tf.Session() as sess:
#   # Task 2: Simulate 10 throws of two dice. Store the results
#   # in a 10x3 matrix.
#
#   # We're going to place dice throws inside two separate
#   # 10x1 matrices. We could have placed dice throws inside
#   # a single 10x2 matrix, but adding different columns of
#   # the same matrix is tricky. We also could have placed
#   # dice throws inside two 1-D tensors (vectors); doing so
#   # would require transposing the result.
#   dice1 = tf.Variable(tf.random_uniform([10, 1],
#                                         minval=1, maxval=7,
#                                         dtype=tf.int32))
#   dice2 = tf.Variable(tf.random_uniform([10, 1],
#                                         minval=1, maxval=7,
#                                         dtype=tf.int32))
#
#   # We may add dice1 and dice2 since they share the same shape
#   # and size.
#   dice_sum = tf.add(dice1, dice2)
#
#   # We've got three separate 10x1 matrices. To produce a single
#   # 10x3 matrix, we'll concatenate them along dimension 1.
#   resulting_matrix = tf.concat(
#       values=[dice1, dice2, dice_sum], axis=1)
#
#   # The variables haven't been initialized within the graph yet,
#   # so let's remedy that.
#   sess.run(tf.global_variables_initializer())
#
#   print(resulting_matrix.eval())
#

# with tf.Graph().as_default():
#   # Create a six-element vector (1-D tensor).
#   primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)
#
#   # Create another six-element vector. Each element in the vector will be
#   # initialized to 1. The first argument is the shape of the tensor (more
#   # on shapes below).
#   ones = tf.ones([6], dtype=tf.int32)
#
#   # Add the two vectors. The resulting tensor is a six-element vector.
#   just_beyond_primes = tf.add(primes, ones)
#
#   # Create a session to run the default graph.
#   with tf.Session() as sess:
#     print(just_beyond_primes.eval())

import tensorflow as tf

# 创建变量 W 和 b 节点，并设置初始值
W = tf.Variable([.1], dtype=tf.float32)
b = tf.Variable([-.1], dtype=tf.float32)
# 创建 x 节点，用来输入实验中的输入数据
x = tf.placeholder(tf.float32)
# 创建线性模型
linear_model = W * x + b

# 创建 y 节点，用来输入实验中得到的输出数据，用于损失模型计算
y = tf.placeholder(tf.float32)
# 创建损失模型
loss = tf.reduce_sum(tf.square(linear_model - y))

# 创建 Session 用来计算模型
sess = tf.Session()

# 初始化变量
init = tf.global_variables_initializer()
sess.run(init)

# 创建一个梯度下降优化器，学习率为0.001
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

# 用两个数组保存训练数据
x_train = [1, 2, 3, 6, 8]
y_train = [4.8, 8.5, 10.4, 21.0, 25.3]

# 训练10000次
for i in range(10000):
    sess.run(train, {x: x_train, y: y_train})

# 打印一下训练后的结果
print('W: %s b: %s loss: %s' % (sess.run(W), sess.run(
    b), sess.run(loss, {x: x_train, y: y_train})))