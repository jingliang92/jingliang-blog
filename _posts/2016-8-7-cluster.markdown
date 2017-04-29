---
layout:     post
title:      "聚类小总结"
subtitle:   "聚类"
date:       2016-8-7 12:00:00
author:     "jingliang"
header-img: "/img/gakki1.jpg"
tags:
    - 聚类
---


## 样本单元间关联的度量

#### 连续变量样本单元间的常用度量：

**首先定义标准符号:**

* $$X_{ik}$$ = 变量K中第i个样本单元的值。
* n = 样本单元的个数。
* p = 变量的个数。

**欧几里得距离(Euclidean Distance):**

$$d({X_i, X_j}) = \sqrt{\sum_{k=1}^{p}(X_{ik} - X_{jk})^2}$$

**闵可夫斯基距离(Minkowski Distance):**

$$d({X_i, X_j}) = \left[\sum_{k=1}^{p}|X_{ik}-X_{jk}|^m\right]^{1/m}$$

**Canberra Metric:**

$$d({X_i, X_j}) = \sum_{k=1}^{p}\frac{|X_{ik}-X_{jk}|}{X_{ik}+X_{jk}}$$

**Czekanowski Coefficient:**

$$d({X_i, X_j}) = 1- \frac{2\sum_{k=1}^{p}\text{min}(X_{ik},X_{jk})}{\sum_{k=1}^{p}(X_{ik}+X_{jk})}$$

以上度量中值越小表示两个对象越相似。

另外也可以自己设计度量方法，只要满足以下条件即可：

1. 对称性：$$d({X_i, X_j}) = d({X_j, X_i})$$
2. 非负性：$$d(X_i, X_j) > 0$$ if $$X_i \ne X_j$$
3. $$d(X_i, X_j) = 0$$ if $${X_i} = {X_j}$$
4. 三角不等式：$$d(X_i, X_k) \le d(X_i, X_j) +d(X_j, X_k)$$

#### 二元变量样本单元间的常用度量:

对样本单元i和j根据每个变量下的0/1情况制作1-1，1-0，0-1，0-0频率表：


|        |     |  单元 |   j  |         |   
|--------|-----|-------|------|---------|
|        |     |   1   |   0  |  total|
| 单元i  |  1  |   a   |   b  |  a+b|  
|        |  0  |   c   |   d  |  c+d |
|        |total| a+c   | b+d  | p=a+b+c+d|  


![](https://onlinecourses.science.psu.edu/stat505/sites/onlinecourses.science.psu.edu.stat505/files/lesson12/formula_09.gif) 

## 不同簇间的度量：

#### 分层的方法：

分为自顶向下和自下到上，自下到上比较常用，即初始将每个单元都看作一个簇，每次迭代将最近的两个簇 合到一起。

1. Single Linkage：$$d_{12} = \displaystyle \min_{i,j}\text{ } d(X_i,Y_j)$$,即两个簇中单元间的最小距离作为判断两个簇是否需要合并的度量,选最小。
2. Complete Linkage：$$d_{12} = \displaystyle \max_{i,j}\text{ } d(X_i, Y_j)$$，即两个簇中单元间的最大距离作为判断两个簇是否需要合并的度量,选最小。
3. Average Linkage：$$d_{12} = \frac{1}{kl}\sum_{i=1}^{k}\sum_{j=1}^{l}d(X_i, Y_j)$$, 即两个簇中单元间所有距离的均值作为判断两个簇是否需要合并的度量,选最小。
4. Centroid Method：$$d_{12} = d(\bar{x},\bar{y})$$，即两个簇平均向量的距离作为判断两个簇是否需要合并的度量,选最小。  

这些聚类方法并无最好之分，在实际中需要多次尝试，选最优。

#### 非分层的方法： 最常用的是k-means聚类。

#### Ward’s Method：
聚类成型应该为椭圆。

令 $$X_{ijk}$$ 表示在簇i中变量k下的第j个观测值的值。

定义如下：

* Error Sum of Squares: $$ESS = \sum_{i}\sum_{j}\sum_{k}{X_{ijk} - \bar{x}_{i\cdot k}}^2$$ . 值越小，表示簇内越紧密。
* Total Sum of Squares: $$TSS = \sum_{i}\sum_{j}\sum_{k}{X_{ijk} - \bar{x}_{\cdot \cdot k}}^2$$ .值越大，簇间越分离。
* R-Square: $$r^2 = \frac{\text{TSS-ESS}}{\text{TSS}}$$.

初始将每个单元都当成一个簇，第一步合并其中的两个，计算ESS和R-Square，选择ESS最小或R-Square最大的两个单元合并成簇。第二步，生成第二个含有两个单元的簇或生成一个含有三个单元的簇。以此类推，直到所有的单元合成一个簇。