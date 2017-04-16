---
layout:     post
title:      "向caret包中的train添加xgboost-R语言"
subtitle:   "利用train实现xgboost的grid search"
date:       2017-3-20 12:00:00
author:     "jingliang"
header-img: "/img/gakki1.jpg"
tags:
    - R语言
---

## 利用事先编好的xgboost调用函数，实现在train中grid搜索调用xgboost算法。

### 首先载入事先编写好的函数代码

代码放在[github上](https://github.com/jingliang92/xgboost_trian)

```    
source("where you put the xgboost.R in your computer")
```   

### 然后设置参数

设置参数有两种，一种是自己定义参数值，另一种是随机生成参数。

定义参数值：

```   
xgbgrid <- expand.grid(eta=0.26, gamma=0, max_depth=seq(3,10,1),
                       min_child_weight=seq(1,12,1), max_delta_step=0, subsample=0.8,
                       colsample_bytree=1, colsample_bylevel = 1, lambda=0.1,
                       alpha=0, scale_pos_weight = 1, nrounds=5, eval_metric='rmse',
                       objective="reg:linear")
```   

随机生成参数：设置 tuneLength 的值。

参数的含义可以在[xgboost文档中](http://xgboost.readthedocs.io/en/latest/parameter.html#parameters-in-r-package)

>注意：每次grid训练最好只看两个或一个变量的grid，如果多个grid可能会出现grid数据大于训练数据的情况，又或是训练时间过长。

### 最后在train中使用：

```   
xgb_model <- train(x, y, method = xgboost, tuneGrid=gbmgrid)
``` 

### 例子：

```   
library(caret)
library(mlbench)
source("D:/xgboost_train/xgboost.R")
data(BostonHousing)

head(BostonHousing)
str(BostonHousing)
BostonHousing$chas <- as.numeric(BostonHousing$chas)

dim(BostonHousing)

xgbgrid <- expand.grid(eta=0.26, gamma=0, max_depth=seq(3,10,1),
                       min_child_weight=seq(1,12,1), max_delta_step=0, subsample=0.8,
                       colsample_bytree=1, colsample_bylevel = 1, lambda=0.1,
                       alpha=0, scale_pos_weight = 1, nrounds=5, eval_metric='rmse',
                       objective="reg:linear")

lm_model <- train(BostonHousing[, 1:13], BostonHousing[, 14], method = 'lm')

xgb_model <- train(BostonHousing[, 1:13], BostonHousing[, 14], method = xgboost)

varImp(xgb_model)
``` 

>注意：这里的train中的xgboost只接受data.frame格式的数值型数据，另外也可以使用train中或caret中的其他函数，例如查看重要变量的VarImp和交叉验证的trControl等。