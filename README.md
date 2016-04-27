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

TensorFlow 中关于CNN的两个重要的函数是 tf.nn.conv2d 和 tf.nn.max_pool ,下面先简单介绍一下函数的用法:

[tf.nn.conv2d](https://www.tensorflow.org/versions/r0.8/api_docs/python/nn.html#conv2d)的参数主要有input、filter 、strides 和 padding。 其中input的大小一般是一个4-D的张量  [batch, in_height, in_width, in_channels]，现在假设输入的样本是图片， batch表示输入的样本数,in_height和in_width 分别表示图片的高和宽，in_channels表示卷积网络中的层数，如果输入的是一幅灰度图片，则in_channels为1，如果是RGB图片则是3。

filter 也是一个4-D的参数[filter_height, filter_width, in_channels, out_channels]，前两个分别是高和宽，这里的in_channels 应该与input 的in_channels的值相同，out_channels是filter的个数，也是最终input 经过与filter卷积之后产生的activation-map的个数

strides 是一个1-D的类型为整型的参数，其长度为四，分别对应着input 中四个维度的sliding window 的步长，我们通常说的stride就是单个filter 在input 上滑动的步长，分别对应着strides[1] 和 strides[2]，表示横向和纵向移动。而strides[0]和strides[3]一般会设为1，表示在每个样本和每个channel 上都进行卷积操作.

padding 在这里是一个字符串，是“SAME”和“VALID”中的一个，分别对应着两个padding 算法，SAME 表示卷积的output的大小与原输入大小相同，用的是zero-padding，VAlID 表示 no-padding，它输出的activation-map的大小是((N-F)/stride) + 1,N表示输入的图片的长度（一般是正方形）,F表示filter的长度，stride是步。

[tf.nn.max_pool](https://www.tensorflow.org/versions/r0.8/api_docs/python/nn.html#max_pool)是一种pooling算法，主要参数是value、ksize、strides 和 padding

value 是一个四维的数组[batch, height, width, channels],  ksize 是一个大于等于四的一维数组，其值表示在对应维度上的窗口，一般情况下ksize[0]和ksize[3]都等于1
参数strides与padding 与 tf.nn.conv2d 中的参数同理。


先定义一些基本函数:

```python
#weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
```

```python
# convolution,这里的x和W都要reshape成4-D的数组
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
```

```python
# pooling
# [[0,3],
#  [4,2]] => 4

# [[0,1],
#  [1,1]] => 1

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
```

现在假设输入的X 大小为 40000*784,40000是样本数，784是一个图片所拉成的向量的长度，需要把它reshape成[40000,28,28,1],可以利用Tensorflow的内置函数
```python
image = tf.reshape(x,[-1,28,28,1])
```

第一个卷积层：

下面初始化filter的大小，假设filter 的长和宽都是5，in_channels 是1，out_channels是32
```python
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
```
激活函数选择了Relu
```python
h_conv1 = tf.nn.relu(conv2d(image,W_conv1)+b_conv1)
#print (h_conv1.get_shape()) # => (40000, 28, 28, 32)

#pooling
h_pool1 = max_pool_2x2(h_conv1)
#print (h_pool1.get_shape()) # => (40000, 14, 14, 32)
```

第二个卷积层：


