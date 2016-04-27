# TFstudyNote
#一.Introduction
[TensorFlow](https://www.tensorflow.org) 是谷歌开源的一个数值计算软件包，它把整个计算过程看作一个数据流图，图中的节点表示各种操作，图中的边表示操作的输入值，一般是张量（多维数组），这就是TensorFlow这个名字的由来，表示数据流在图中的传播，TensorFlow类似于Caffe、MXNet ，也是一种深度学习包。TensorFlow 采用的是一种声明式的语言，即它可以先定义网络的结构，在声明阶段不用关心各个张量的数值，类似于占位符。当网络声明完成之后只需要给定训练数据，就可以自动完成整个网络的训练过程。

#二.Instance
TensorFlow官网给定的是一个利用CNN训练手写字符数据集（MNIST）的[入门教程](https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html)，下面我将结合这个入门教程和[Kaggle](https://www.kaggle.com/c/digit-recognizer)中的字符识别竞赛介绍TensorFlow利用CNN在MNIST数据集上的具体应用

##建立一个Softmax 回归模型
1.Start Session
```python
#load tensorflow assume the tensorflow packages has been installed on your pc
import tensorflow as tf
sess = tf.InteractiveSession()
```
Tensorflow 为了保证高效率，其后台采用了C++作为开发语言，而python只是用作与用户进行交互的语言，用户用python在前台构建的Graph 将通过Session与后台交互，从而实现高效率的计算. tf.Seesion() 与 tf.InteractiveSession() 的不同之处在于，前者一旦构建之后就不可以在后面改图结构了(未验证)，而后者与Ipython类似用户可以交互地修改与执行

2.Build Graph

1)PlaceHolders
```python
#MNIST的特征数784个，shape的第一个参数是None表示对其不限制，依传入的实参而定，一般为Batch_size（一次训练的样本数）
#y_的第二维为10 表示共有10个类，使用0-1编码表示，即若为第i类，则y_[i]=1，其他为0
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
```
[tf.placeholder(dtype, shape=None, name=None)](https://www.tensorflow.org/versions/r0.8/api_docs/python/io_ops.html#placeholder)可以看作在声明一个多维数组，它现在并没有具体的值，它的值需要在训练或测试时利用Session.run()、Tensor.eval() 等方法通过 参数 feed_dict 传递进来，后面将会利用到 feed_dict这个参数。
    
2)Variables

```python
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
```
Variabels 一般用来表示在网络中需要更新的参数，一般都需要初始化

```python
#初始化网络中所有变量
sess.run(tf.initialize_all_variables())
```

3）Predicted Class and Cost Function

```python
y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
```
[tf.nn.softmax](https://www.tensorflow.org/versions/r0.8/api_docs/python/nn.html#softmax) 的参数的大小为[BATCH_SIZE,NUM_CLASSES]，结果形式可参见上述API-DOC。
[tf.reduce_mean](https://www.tensorflow.org/versions/r0.8/api_docs/python/math_ops.html#reduce_mean)计算数组在指定维度的均值
[tf.reduce_sum](https://www.tensorflow.org/versions/r0.8/api_docs/python/math_ops.html#reduce_sum)与reduce_mean同理。不过cross_entropy 应该直接等于 -tf.reduce_sum(y_ * tf.log(y)),再加一个tf.reduce_mean 也不影响结果。

4)Train the Model

```python
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
```

注意train_step.run中的参数feed_dict ,这里以字典的形式传入了两个placeholder的值。

5）Evaluate The Model

```python
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```
注意最后一行的accuracy.eval()一般等价于tf.get_default_session().run(accuracy),即使用默认的Session运行，但是当有多个Session时（可以看作多个网络）一般需指定使用的Session即使用sess.run()的形式。[参考](https://www.tensorflow.org/versions/r0.8/resources/faq.html#contents)。accuracy.eval()表示在给定参数feed_dict 的情况下执行完整个图之后返回所要求的值accuracy，


##建立一个CNN模型

