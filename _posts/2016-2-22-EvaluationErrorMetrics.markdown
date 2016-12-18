---
layout:     post
title:      "7 Important Model Evaluation Error Metrics Everyone should know-翻译"
subtitle:   "模型误差度量的几个方法"
date:       2016-02-22 12:00:00
author:     "jingliang"
header-img: "/img/gakki.jpg"
tags:
    - 数据分析
---

# 七个重要的模型误差度量
   你的目的不仅仅是构建一个预测模型，而是创建和选择一个对样本以外的数据同样具有高度精度的模型。因此在用模型计算预测值之前，去检测模型准确度是非常重要的一个步骤。

## 目录：
1. 混淆矩阵（Confusion Matrix）
2. 增益和提升图（Gain and Lift Chart）
3. K-S图（Kolmogorov Smirnov Chart）
4. AUC – ROC
5. 基尼系数（Gini Coefficient）
6. Concordant – Discordant Ratio
7. 均方根误差（Root Mean Squared Error）
8. 交叉验证（Cross Validation (Not a metric though!)）

## 提醒：预测模型的类型
回归模型（输出是连续的）和分类模型（输出是标称数据[离散数据]或二分数据）具有不同的评估度量。

### 分类问题：
1. 类别输出：SVM和KNN创建一个分类输出。例如，在二分类问题中，输出可能是0或是1。但是，现在有一些算法可以将类别输出转换为概率。然而这些算法在统计学界是不太被认同的。
2. 概率输出：逻辑回归、随机森林、梯度提升（Gradient-Boosting）、Adaboost是概率输出的。转换概率输出到类别输出只是创建一个阈值概率的问题。

### 在回归问题中，输出总是连续的不需要再进一步处理。

## 1. 混淆矩阵（Confusion Matrix）
混淆矩阵是一个N X N的矩阵，N是预测类别的数目。如下表所示：

Model&Target |      Positive      |     Negative        
-------------|--------------------|---------------------
Positive     | True Positives(TP) | False Positives(FP) 
Negative     | False Negatives(FN)| True Negatives(TN)  
tota;        | Positive Samples(P)| Negative Samples(N)  

* Accuracy = (TP+TN)/(P+N) ：预测正确的数目占总数的比值
* Positive Predictive Value = Precision = TP/(TP+FP)：模型预测Positive中正确占比
* Negative Predictive Value = TN/(TN+FN)：模型预测Negative中正确占比
* True Positive Rate = Recall = Sensitivity = TP/P：在实际Positive例子中模型预测正确占比
* True Negative Rate = Specificity = TN/N ：在实际Negative例子中模型预测正确占比

这里用R中ROCR例子做代码示例(与原文不同）：[代码参考](http://iccm.cc/classification-model-evaluation-confusion-matrix/)


##### 数据展示，其中prediction是预测值，labels为真实值。
```{r}
library(ROCR)
data(ROCR.simple)
data <- as.data.frame(ROCR.simple)[1:10, ]
```

#### 设置阀值概率，这里设置为0.5，即predictions>0.5为1，否则为0。
```{r}
pred.class <- as.integer(ROCR.simple$predictions > 0.5)
print(cft <- table(pred.class, ROCR.simple$labels))
```

#### 使用caret包中的confusionMatrix函数计算混淆矩阵
```{r}
library(caret)
#设置positve为1
confusionMatrix(cft, positive = "1")
#confusionMatrix(pred.class, ROCR.simple$labels, positive = "1")
```
例子中的准确性为85%。从上表可以看出，阳性(positive)预测值很高，阴性(negative)预测值也很高。灵敏度(Sensitivity)很高，特异性(Specificity)也比较高。这主要是由于我们所选的门槛值造成的。如果我们增大门槛值，两对值差距会变大。

一般，我们考虑上面度量中的某一个。例如，在一个制药公司中，他们更关心最小错误阳性诊断(1-Specifiicty)。因此，他们更关心高特异性。另一个方面，一个损耗模型会更关心灵敏度。混淆矩阵一般只被用在类别输出模型。

## 2. 增益和提升图（Gain and Lift charts）
增益和提升图主要是检查概率的排序。下面是建立增益和提升图的步骤：
1. 计算每个观察值的概率
2. 降序排列这些概率
3. 构建十分位数分析，即将观测值以每10%的数目分一组。
4. 计算每个分组内的反应率和累积反应率（Good (Responders) ,Bad (Non-responders) and total）

注：[关于Gain and lift这篇讲解的比较清楚](http://f.dataguru.cn/thread-535331-1-2.html)

##### 原文中的表如下，可以据此画出提升图：

![alt text](http://i1.wp.com/www.analyticsvidhya.com/blog/wp-content/uploads/2015/01/LiftnGain.png)

这是一个非常有用的表。累积增益图是 Cumulative %Right 和 Cummulative %Population之间的图。图如下所示：

![alt text](http://i0.wp.com/www.analyticsvidhya.com/blog/wp-content/uploads/2015/01/CumGain.png)

这个图告诉你，你的模型是如何将反应人数和非反应人数区分的。例如，第一个十分位数有整体的10%，有14%的反应人数。这意味着我们在第一个十分位数有140%的提升。

我们在第一个十分位数所能达到的最大提升是什么？从上面的第一张表，我们知道总共的反应人数为3850.第一个十分位数包含543个观测值。因此，在第一个十分位数的最大提升为543/3850~14.1%。因此，我们非常接近完善这个模型。

现在让我们画提升曲线。提升曲线是提升totallift和各十分位数%population间的关系曲线图。注意对一个随机模型，这里总保持100%的平稳。下面是这个例子的图：

![alt text](http://i0.wp.com/www.analyticsvidhya.com/blog/wp-content/uploads/2015/01/Lift.png)

你也可以画出相对提升图（decile wise lift with decile number ）：

![alt text](http://i2.wp.com/www.analyticsvidhya.com/blog/wp-content/uploads/2015/01/Liftdecile.png)

这个图告诉我们什么？它告诉我们直到第七个分位数模型表现的很好。每个十分位数将向无反应者偏斜。任何模型 lift @ decile在100%上（最小到第三十分位数，最大到第七十分位数）是一个好的模型，否则你可能要考虑先进行抽样。

增益提升图被广泛用在有针对性活动的问题上。它告诉我们在哪个部分我们可以为某个具体的活动定位客户。同样，它告诉你在新的目标基准上你的期望回应是多少？

这里用R中ROCR中的例子演示[代码参考](http://iccm.cc/classification-model-evaluation-gain-chart-lift-chart/)：

#### 数据提取：
```{r}
library(ROCR)
data(ROCR.simple)
data <- as.data.frame(ROCR.simple)[1:10, ]
```

#### 排序预测分数，计算所有预测中，预测为正的比例(rpp)，即随机模型反应率；计算预测为正的事件中，正确的概率(tpr)，即模型的反应率。

```{r}
data <- data[order(data[, 1], decreasing = TRUE), ]
data$rpp <- row(data[, 1, drop = FALSE])/nrow(data)
data$target_cum <- cumsum(data[, "labels"])
data$tpr <- data$target_cum/sum(data[, "labels"])
data$lift <- data$tpr/data$rpp
data
```

#### 图形展示：
```{r}
par(mfrow = c(1, 2))
plot(data$rpp, data$tpr, type = "l", main = "Gain Chart")
plot(data$rpp, data$lift, type = "l", main = "Lift Chart")
```

#### 利用ROCR包简单作图：
```{r}
require(ROCR)
data(ROCR.simple)
pred <- prediction(ROCR.simple$predictions, ROCR.simple$labels)
par(mfrow = c(1, 2))
gain <- performance(pred, "tpr", "rpp")
plot(gain, main = "Gain Chart")
lift <- performance(pred, "lift", "rpp")
plot(lift, main = "Lift Chart")
```

##  3. K-S图
K-S图衡量分类模型的表现。更准确的说，K-S是the positive and negative分布之间分离程度的衡量。如果K-S 是100，那么 按分数将整体分成两组，一组包含全部的positive，一组包含全部的negative。
另一方面，如果模型不能区分the positive and negative，那么就像模型从总体中随机选择例子，则K-S会是0。在大多数分类模型中，K-S会落在0到100之间。K-S值越高，模型的分离程度越好。

原文中例子如下表：

![alt text](http://i0.wp.com/www.analyticsvidhya.com/blog/wp-content/uploads/2015/01/KS.png)

我们也可以画%Cumulative Good and Bad看最大的分离。下面是一个简单图：

![alt text](http://i0.wp.com/www.analyticsvidhya.com/blog/wp-content/uploads/2015/01/KS_plot.png)

上面这些度量主要用于分类问题，下面让我们学习新的重要的度量。

##  4. ROC曲线下面积
这是另一个在工业界流行使用的度量。使用ROC最大的优点是在反应者的比例里是独立变化的，即样本中positive比例的变化不会引起ROC曲线的变化，不同的样本可能具有不同的比例。下面的部分会让你清楚这句表述。

让我们先理解什么是ROC曲线，如果我们看下面的混淆矩阵，我们观察一个概率模型，为每个度量获取不同的值。

![alt text](http://i0.wp.com/www.analyticsvidhya.com/blog/wp-content/uploads/2015/01/Confusion_matrix.png)

因此，对于每个灵敏度sensitivity，我们获得一个不同的specificity。下面是两个不同的曲线：

![alt text](http://i2.wp.com/www.analyticsvidhya.com/blog/wp-content/uploads/2015/01/curves.png)

这个ROC曲线是灵敏度sensitivity和（1- specificity）画成的。(1-specificity)是false positive rate and ，sensitivity是True Positive rate。下面是ROC图：

![alt text](http://i2.wp.com/www.analyticsvidhya.com/blog/wp-content/uploads/2015/01/ROC.png)

让我们设置门槛值为0.5，下面是混淆矩阵：

![alt text](http://i0.wp.com/www.analyticsvidhya.com/blog/wp-content/uploads/2015/01/Confusion_matrix2.png)

正如你所看到的，在这个门槛值下灵敏度是99.6%，(1- specificity)是60%。这个坐标点在我们的ROC曲线上。把这个曲线变成到一个单一的数字，我们定义在ROC曲线下的面积，称为AUC。

注意整体的面积是1。因此AUC是曲线面积和整体面积的比。这个例子的AUC是96.4%。下面是一些经验法则：

* .90-1=excellent(A)
* .80-.90=good(B) 
* .70-.80=fair(C) 
* .60-.70=poor(D) 
* .50-.60=fail(F) 

在当前模型下，我们处在excellent。但这可能只是过度拟合。因此在这个例子中用交叉集验证是非常重要的。

需要记住的要点：
1. 作为类别输出的模型，在ROC曲线上将被呈现为一个单点。
2. 2.这种模型不能互相比较，作为判断需要采取一个单一的指标而不是多个指标。一个参数是（0.2,0.8）的模型和参数是（0.8,0.2）的模型是同一个模型，因此这些指标不能被直接比较。
3. 3.在概率模型下，我们是足够幸运的去获得一个单一的数字（AUC-ROC）。但是我们需要看整体的曲线获得令人信服的决定。也可能在某些区域某个模型表现的更好，在其他区域另一个模型表现的更好。？

注：[ROC的优点](http://alexkong.net/2013/06/introduction-to-auc-and-roc/)

使用ROC而不使用提升曲线的原因:

lift 依赖于total response rate of the population，因此如果total response rate of the population改变，同样的模型将会得到不同的提升表。ROC曲线依赖于sensitivity和specificity，这两个变化是同步的，因此不会影响ROC曲线的变化。

这里用R中ROCR中的例子演示[代码参考](http://iccm.cc/classification-model-evaluation-roc-chart-auc/)：

#### 计算false positive rate(1-specificity)【FPR】，true positive rate(sensitivity)【TPR】 
```{r}
data(ROCR.simple)
data <- as.data.frame(ROCR.simple)[1:10, ]
data <- data[order(data[, 1], decreasing = TRUE), ]
data$target_cum <- cumsum(data[, "labels"])
data$tpr <- data$target_cum/sum(data[, "labels"])
data$fpr <- (row(data[, 1, drop = F]) - data$target_cum)/(nrow(data) - sum(data[, 
    "labels"]))
data
```

#### 图形展示
```{r}
plot(data$tpr, data$fpr, type = "l", main = "ROC Chart") 
```

#### 用ROCR包画图、计算AUC
```{r}
pred <- prediction(ROCR.simple$predictions, ROCR.simple$labels)
roc <- performance(pred, "tpr", "fpr")
plot(roc, main = "ROC chart")
auc <- performance(pred, "auc")@y.values
auc
```

## 5. 基尼系数（Gini Coefficient）

基尼系数有时被用在分类问题。基尼系数可以直接从AUC ROC推得。基尼系数需要AUC值。
Gini = 2*AUC – 1

基尼系数大于60%是一个比较好的模型。在这个例子中基尼系数是92.7%。

## 6. Concordant–Discordant ratio
这是另一个分类模型中的重要度量。为了去理解这个指标，我们假设有三个学生今年可能通过。下面是我们的预测：
A-0.9，B-0.5，C-0.3

如果我们从这些学生中取对，可以得到AB，BC，CA。现在，一年后，A和C通过了，B失败了。现在我们选出一个失败一个成功的对，可以得到AB和BC。这些concordant-pair成功的概率大于失败的概率。而discordant pair是相反的。如果概率相同，我们说它是a tie。让我们看下在我们的例子中：
AB-Concordant, BC-Discordant

因此，在这个例子中Concordant是50%，超过60%我们认为是一个好的模型。这个度量一般不会用在决定有多少目标客户。它主要用于评估模型的预测能力。可以用KS、提升图去决定有多少目标客户。

##  7. 均方根误差（Root Mean Squared Error (RMSE)）  
均方根误差是回归模型中最流行的评价指标。它遵循一个假设，误差是无偏的且服从正态分布。下面是均方根误差的关键点：
1. 平方根使这个度量来说明大量的偏差。
2. 这个度量的平方性质帮助提供更多的鲁棒性，可以消除正和负的误差值。换句话说，这个度量贴切的显示出误差项的合理大小。
3. 它避免使用绝对误差值，绝对误差值在数学计算中是不可取的。
4. 当我们有更多样本，使用均方根误差重建误差分布被认为更有可靠性。
5. 均方根误差受异常值影响较大。因此，在使用这个度量前请确定在你的数据集中删除了异常值。
6. 相比平均绝对误差，均方根误差给予更好的权重和能够惩罚大误差

均方根误差公式为：

![alt text](http://i1.wp.com/www.analyticsvidhya.com/wp-content/uploads/2016/02/rmse.png?w=358)

N是观察总数。

除了这7个指标，还有另一种方法来检查模型的性能。这七种方法是统计学显著的。但是，随着机器学习的到来。我们现在有更稳健的方法来进行模型的选择。它就是交叉验证。虽然交叉验证不是一个真正的评价指标，但它被广泛用来沟通模型的准确性。但是，交叉验证的结果为概括模型的性能提高了一个很好的直观的结果。

##  8. 交叉验证   
交叉验证在任何种类的模型中都是最重要的概念之一。简单的说，就是将训练集再次分成训练集和测试集。

![alt text](http://i0.wp.com/www.analyticsvidhya.com/wp-content/uploads/2015/05/validation.png)

上图展示如何用快速样本验证模型。我们简单的将样本分为两个样本，其中一个建立模型。另一个样本用来快速验证。

上面的方法有消极的一面吗？

我认为训练集减少是这种方面的负面影响。因此这种模型具有高度偏差。这不会得到最佳估计的系数。所以最好的选择是什么？

如果我们把训练集分成相等的两份，一份用于训练模型，另一份用于验证。然后我们再在另一份训练，一份验证。这种方法我们整体上训练模型，但是一次训练一半。这减少偏差，因为一定程度上的样本选择给了一个更小的模型训练模型。这种方法被称为2倍交叉验证。
####  K折交叉验证
我们从2倍交叉验证扩展到k倍交叉验证。现在，我们可视化k折交叉验证的工作过程。

![alt text](http://i2.wp.com/www.analyticsvidhya.com/wp-content/uploads/2015/05/kfolds.png)

这是7折交叉验证。

这里是过程：将整体分成7个相等大小的样本。现在我们训练其中的6个，剩下的一个用于验证。然后，我们选择不同的验证集，迭代这个过程7次。在这个过程中，我们在每个样本上训练模型，并在每个样本上做了验证。这是一种在预测中减少选择偏差和方差的方法。一旦我们有了七个模型，我们利用平均误差选择最好的模型。

它是如何发现最好（无过拟合）模型的？

K折交叉验证是广泛被使用来检测模型是否过拟合的。如果性能指标在k次模拟中是相互接近的，指标的均值是最大。

如何在模型中应用k折交叉验证？

原文给出了python代码，这里给出R代码，利用CARET：

```{r}
library(caret)
data(iris)
TrainData <- iris[,1:4]
TrainClasses <- iris[,5]
fitControl = trainControl(
    method = "repeatedcv",
    number = 2,
    repeats = 2,
    returnResamp = "all")
knnFit1 <- train(TrainData, TrainClasses,
                 method = "knn",
                 preProcess = c("center", "scale"),
                 tuneLength = 10,
                 trControl = fitControl)
summary (knnFit1)
```

如何选择k？

这是个棘手的部分，我们权衡的选择K。

K如果小了，我们会得到高的选择偏差和低方差。

K如果大了，我们会得到低的选择偏差和高方差。

在大多是时候K选为10.

[本文翻译自《7 Important Model Evaluation Error Metrics Everyone should know》](http://www.analyticsvidhya.com/blog/2016/02/7-important-model-evaluation-error-metrics/)