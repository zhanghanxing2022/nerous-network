# nerous-network

[TOC]

This is a simple neural network based on numpy constructed by myeslf.It maybe very rough.
## 目标

<font size = 20>仿照keras建造一个初步的神经网络架构。它将包括</font><br>
<table>
  <tr>
<td>激活函数</td><td>sigmoid，softmax，relu</td>
  </tr>
  <tr>
<td>网络层</td><td>全连接层，2D卷积层，2D最大池化层</td>
    </tr>
  <tr>
    <td>优化器</td><td>SGD随机梯度下降</td>
    </tr>
  <tr>
  <td>正则化</td><td>L2正则化</td>
  </tr>
  <tr>
    <td colspan = 2>模型导出与导入</td>
    </tr>
  </table>
  
  ## 可叠加模型
  
  本次构建尝试使将各个功能相分离，使得各个组件低耦合，高内聚，并且可以给予使用者一些自由，使其可以动手搭建属于自己的神经网络结构，而不限于代码提供的示例。
  
  实现目标如下:<br>
  <br>
  <code>
  input = Tensor(28,28)</code>
  <br>创建输入的张量<br>
  <code>
  out = Convolution2D(filters = 8,keneral=（3,3）,activator='relu')(input)
</code>
<br>
使用卷积层，滤波数量为8，卷积核为3x3，暂不支持stride和pad;name参数可选；可选参数还包括biasUsed。激活函数请不要选softmax
<br>
<code>
  out = MaxPooling2D(shape=（3,3）)(out)
  </code>
  <br>使用池化层，视野为3x3，默认stride与shape相同。暂不支持stride<br>
  <code>
  out = Flatten()(out)
  </code>
  <br>将数据展平<br>
  <code>
  out = Dense(neurons=10,acivator='softmax')(out)
  </code>
  <br>全连接层，可选参数还包括biasUsed，激活函数为softmax<br>
  <code>
  model = Model(input,out)
  </code>
  <br>建立神经网络<br>
  <code>
  optimizer = SGD(lr=0.001, decay=1.0, clipvalue=10)
   </code>
  <br>
  <code>
  model.compileLoss(Cross_Entropy())
   </code>
  <br>
  <code>
  model.compileRegular(L2Regularization(lamd=0.001))
   </code>
  <br>
  <code>
  model.compileOptimizer(optimizer)
  </code>
  <br>
  <code>
  model.fit(x_train, y_train, iteration=4000, filename='./test/2/', xtest, ytest, step=100)
</code>
  <br>出前三个参数，其它可选，当前为使用SGD训练4000次，每训练100次使用测试集测试准确率，loss输出为./test/2/loss.csv,追确率输出为./test/2/acc.csv<br>
  <code>
path = model.logModel('./test/2/model.h5')
  </code>
  <br>保存模型<br>
  <code>
model.load_model(path)
  </code>
  <br>载入模型<br>
  <code>
model.evaluate(x_test, y_test_onehot)#
  </code>
  <br>测试,返回准确率<br>

  
