# 一.创建会话
## 1.创建一个会话并运行
```python
sess1 = tf.Session(graph = g1)
with sess1.as_default():
    tensor.eval()
```
或
```python
with tf.Session(graph = g1) as sess:
    sess.run(tensor)
    sess.run([tensor1, tensor2...])
```
所有节点的运行都会返回一个tensor，该tensor是节点输出的结果。在python中是numpy.ndarray类型的数据。我们可以print查看结果，或者用一个变量名记录。用变量名记录后，可以对其进行其它的不属于tf的运算。达到tf和py其余库混用的效果。
# 二.会话的深入理解以及高级用法
## 1.会话的深入理解
由于之前定义好了图，我们使用会话计算一个节点的输出值时，它会按定义好的图递归地计算用到的之前节点的值。事实上，步骤应该为，先递归确定好用到的节点，然后对这些节点分配计算资源，然后计算。计算完毕后，释放计算资源。
