# 一.定义计算图
## 1.定义计算图
```python
g1 = tf.Graph()
with g1.as_default():
    ....
```
# 二.在计算图中增加节点
## 1.节点，张量与计算
图上的每一个节点一定对应一个计算，也对应计算出来的张量，张量在节点传递形成流。这也是TensorFlow的命名原因。  
我们在计算图中定义节点，它在图中是一个计算，在会话中会得到计算的张量。一个定义，其实是对应三个东西。这也是容易引起混淆的地方，应注意细微的差别。    
一个节点主要有三个属性：名字（name），维度（shape）和类型（type）。  
名字是作为图中的一个唯一标识符，它可以缺省由内部自动分配。要区分名字和定义时给节点的变量名二者的差异。变量名是一个指针的作用，可以有多个变量名指向一个节点。但节点的名字只有一个。  
维度描述了节点计算后得到的张量的维度信息。  
类型描述了节点处理数据的类型。tf是不会自动帮你转化类型的。要显式转化。不同类型的张量是不可以做计算的。    
## 2.节点的类型
节点可大致分为两类，一类是初始节点。这些节点没有输入，只有输出。这些节点也没有定义计算。其实就是初始化一些常量，变量等。另一类便是有输入，有输出的节点，这些节点定义了一些计算，并对应了输出的张量。  
### 2.1中间节点
一张图一定有一些初始的节点，主要的类型如下：
```python
tf.constant()
a = tf.constant([1, 2], name = 'a', dtype = tf.float32)
name和dtype可以缺省。
```
```python
tf.Variable()
a = tf.Variable([1, 2], name = 'a')
name可以缺省。注意Variable是可变类型的。
注意定义变量后，要用tf.global_variables_initializer()来进行初始化操作。
```
```python
tf.placeholder(dtype, shape=None, name=None)
dtype必须给出，shape和name不一定给出。shape若不给出，会自动根据输入数据得出。若提前给出，则输入数据和shape不符合时会报错。name不给出时会自动生成唯一的name。
如果在会话中计算节点时，如果计算该节点递归发现需要placeholder，那么必须以字典的形式给出。
x = tf.placeholder(tf.float32, shape = (3,2), name = 'input')
sess.run(y, feed_dict = {x:[[0.7,0.9],[0.1,0.4],[0.5,0.8]]})
其中字典中的键值对中的值可以为：Python scalar, string, list, or numpy ndarraythat can be converted to the same `dtype` as that tensor。
```
注意只有初始节点能保留值，其余节点是不会保留值的。其余节点定义的是运算，也许会产生中间变量，但这些变量并不可见，也不可更改。
### 2.2计算节点
计算节点有很多，如重载运算符+-*/，以及一些函数形式的：tf.matmul(a, b)。不再列举。
# 三.节点使用心得
## 1.节点的name属性的作用
考虑如下代码：
```python
import tensorflow as tf
g1 = tf.Graph()
with g1.as_default():
    a = tf.Variable([[1,2]])
    b = tf.Variable([[2],[3]])
    c = tf.matmul(a,b)
    c = a * c
    init_op = tf.global_variables_initializer()
with tf.Session(graph = g1) as sess:
    sess.run(init_op)
    print(sess.run(c))
```
当我们运行sess.run(c)时，tf是怎么确定是run的哪个c呢？甚至，如果画成计算图的话，c是不是形成一个绕自己的环了吗？会无限循环吗？  
事实上，不会形成环。如果觉得形成环，那便是混淆了name属性和变量名。c是一个变量名，它是py的，可以随便指向的。  
我们叙述一下代码过程，问题便清楚了。当代码执行到c = tf.matmul(a,b)时，c指向了节点tf.matmul(a,b)，注意，此时tf已经在g1图中为tf.matmul(a,b)分配了唯一一个name属性了，这里a和b的作用相当于一个指针，来连接。当执行到c = a * c时，tf又在g1图中生成了一个a指向的节点和c指向的节点的相乘运算节点，并分配了唯一一个name属性来标识，接着=表示c指向它。  
这里容易让人感到困惑的地方是，在py的运算习惯中，对于c = a * c，c之前的值会被当做垃圾回收掉。但是这里c之前指向的节点不会被回收，只要它被定义了，那它便一直存在于图中，不管有没有py的变量指向它。当然，如果没有变量指向它，就不能简单用sess.run(变量名)来看这个节点的输出值了。  
总之，tf的语法虽然和py不同，但py作为环境，tf还是要服从于py的。严格来说，并不能说tf自成语法，tf无非还是一些python的函数，类等等。也是顺序执行的。只是，tf的函数，类的行为确实和一般的函数，类有所不同。比如计算节点的定义其实就是函数的调用执行，调用的结果就是在计算图上增加节点，并返回该节点的引用。而计算图由g1 = tf.Graph()定义，是一个类。函数可以更改类中的图属性。这便是增加节点的实际实现方式。  
所以，虽然我们在抽象的层面上理解tf可能更直观，更有导向性。但在写代码时，我们还是要想着函数调用，类的初始化等等这些基本的东西来写，这样更严谨，而且不会引起混淆。  
# 四.高级用法
## 1.get_variable和variable域的使用
使用样例
```python
#通过tf.variable_scope函数控制tf.get_variable函数来获取以及创建过的变量
with tf.variable_scope("zyy"):#zyy的命名空间
        #zyy的命名空间内创建名字为v的变量
        v=tf.get_variable("v",[1],initializer=tf.constant_initializer(1.0))  
with tf.variable_scope("zyy"):
        #通过tf.get_variable函数创建v的变量，则会失败，由于在zyy空间中已经生成了一个v的变量
        v=tf.get_variable("v",[1])  
with tf.variable_scope("zyy",reuse=True):
      v1=tf.get_variable("v",[1])
print v==v1   #输出为True
```
官方样本
```python
get_variable(name, shape=None, dtype=None, initializer=None, regularizer=None, trainable=True, collections=None, caching_device=None, partitioner=None, validate_shape=True, use_resource=None, custom_getter
=None)
If initializer is `None` (the default), the default initializer passed in the variable scope will be used. If that one is `None` too, `glorot_uniform_initializer` will be used. The initializer can also be a Tensor, in which case the variable is initialized to this value and shape.
......
```
通过以上使用样例及官方样本，可以总结如下：  
当tf.get_variable用于创建变量时，则与tf.Variable的功能基本相同。但tf.get_variable函数，变量名称是一个必填的参数，它会根据变量名称去创建或者获取变量。二者返回的都是<class 'tensorflow.python.ops.variables.Variable'>类型。可以理解为tf.get_variable是封装的tf.Variable，所以推荐使用tf.get_variable。  
tf.get_variable可以选择初始化方法，但如果用tf.Variable，则还需要手动实现。  
初始化方法大概有如下几种：  
tf.constant_initializer：常量初始化函数  
tf.random_normal_initializer：正态分布  
tf.truncated_normal_initializer：截取的正态分布  
tf.random_uniform_initializer：均匀分布  
tf.zeros_initializer：全部是0  
tf.ones_initializer：全是1  
tf.uniform_unit_scaling_initializer：满足均匀分布，但不影响输出数量级的随机值  
并且tf.variable_scope在reuse=True的情况下可以获取已经创建过的变量。  
如果tf.variable_scope函数使用参数reuse=False（默认为False）创建上下文管理器，则tf.get_variable函数可以创建新的变量。但不可以创建已经存在的变量即为同名的变量。


