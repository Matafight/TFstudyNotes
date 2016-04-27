# TFstudyNote
#一.Introductio
[TensorFlow](https://www.tensorflow.org) 是谷歌开源的一个数值计算软件包，它把整个计算过程看作一个数据流图，图中的节点表示各种操作，图中的边表示操作的输入值，一般是张量（多维数组），这就是TensorFlow这个名字的由来，表示数据流在图中的传播，TensorFlow类似于Caffe、MXNet ，也是一种深度学习包。TensorFlow 采用的是一种声明式的语言，即它可以先定义网络的结构，在声明阶段不用关心各个张量的数值，类似于占位符。当网络声明完成之后只需要给定训练数据，就可以自动完成整个网络的训练过程。

#二.Instance
TensorFlow官网给定的是一个利用CNN训练手写字符数据集（MNIST）的[入门教程](https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html)，下面我将结合这个入门教程和[Kaggle](https://www.kaggle.com/c/digit-recognizer)中的字符识别竞赛介绍TensorFlow利用CNN在MNIST数据集上的具体应用

1.Start Session
```python
#load tensorflow assume the tensorflow packages has been installed on your pc
import tensorflow as tf
sess = tf.InteractiveSession()
```
Tensorflow 为了保证高效率，其后台采用了C++作为开发语言，而python只是用作与用户进行交互的语言，用户用python在前台构建的Graph 将通过Session与后台交互，从而实现高效率的计算. tf.Seesion() 与 tf.InteractiveSession() 的不同之处在于，前者一旦构建之后就不可以在后面改图结构了(未验证)，而后者与Ipython类似用户可以交互地修改与执行

2.Build Grap
1)PlaceHolders
```python
#MNIST的特征数784个，shape的第一个参数是None表示对其不限制，依传入的实参而定，一般为Batch_size（一次训练的样本数）
#y_的第二维为10 表示共有10个类，使用0-1编码表示，即若为第i类，则y_[i]=1，其他为0
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
```
[tf.placeholder(dtype, shape=None, name=None)](https://www.tensorflow.org/versions/r0.8/api_docs/python/io_ops.html#placeholder)可以看作在声明一个多维数组，它现在并没有具体的值，它的值需要在训练或测试时利用Session.run()、Tensor.eval() 等方法通过 参数 feed_dict 传递进来，后面将会利用到这个参数。
    
2)Variables



