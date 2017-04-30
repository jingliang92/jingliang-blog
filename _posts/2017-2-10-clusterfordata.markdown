---
layout:     post
title:      "聚类在模型构建中的应用"
subtitle:   "利用聚类创建新变量"
date:       2017-2-10 12:00:00
author:     "jingliang"
header-img: "/img/gakki1.jpg"
tags:
    - 聚类
---


#### 导入所需包及数据

```
library(caret)
library(data.table)
library(dummies)

load_train <- read.csv('https://datahack-prod.s3.ap-south-1.amazonaws.com/train_file/train_u6lujuX_CVtuZ9i.csv')
load_test <- read.csv('https://datahack-prod.s3.ap-south-1.amazonaws.com/test_file/test_Y3wMUE5_7gLdaTN.csv')
```

#### 查看数据

```
dim(load_train)
str(load_train)
summary(load_train)
```

数据量很小，但大部分变量都存在缺失值。

#### 首先使用gbm建模

```
ctrl <- trainControl(method = "repeatedcv", number = 5, classProbs = TRUE,
                     summaryFunction = twoClassSummary)
                     
gbm_first <- train(load_train[,2:12], load_train[, 13], method = 'gbm', trControl=ctrl,  metric = "ROC")
gbm_first
Stochastic Gradient Boosting 

614 samples
 11 predictor
  2 classes: 'N', 'Y' 

No pre-processing
Resampling: Cross-Validated (2 fold, repeated 1 times) 
Summary of sample sizes: 307, 307 
Resampling results across tuning parameters:

  interaction.depth  n.trees  ROC        Sens       Spec     
  1                   50      0.7339060  0.4270833  0.9810427
  1                  100      0.7344737  0.4375000  0.9620853
  1                  150      0.7277720  0.4635417  0.9123223
  2                   50      0.7187500  0.4322917  0.9715640
  2                  100      0.7188241  0.4479167  0.9170616
  2                  150      0.7163310  0.4687500  0.9146919
  3                   50      0.7280682  0.4479167  0.9336493
  3                  100      0.7344491  0.4739583  0.9218009
  3                  150      0.7307464  0.4791667  0.9146919

Tuning parameter 'shrinkage' was held constant at a value of 0.1
Tuning parameter 'n.minobsinnode' was held constant at a value of 10
ROC was used to select the optimal model using  the largest value.
The final values used for the model were n.trees = 100, interaction.depth = 1, shrinkage = 0.1 and n.minobsinnode = 10.                     
```

看起来似乎不错，查看重要变量：

```
varImp(gbm)
gbm variable importance

                  Overall
Credit_History    100.000
LoanAmount         32.499
ApplicantIncome     9.887
Property_Area       9.539
CoapplicantIncome   7.557
Married             4.962
Dependents          2.792
Loan_Amount_Term    2.717
Education           1.895
Gender              0.000
Self_Employed       0.000
```

其中Credit_History变量非常重要，在前面查看变量中，我们发现Credit_History同样存在缺失值。

将上面所建模型应用于测试集，并以Credit_History作为判断基准进行比较（将test缺失值赋值为1）

```
Loan_Status <- predict(gbm, load_test[,2:12])
result_gbm_first <- data.frame(Loan_ID=load_test$Loan_ID, Loan_Status)

base <- ifelse(load_test$Credit_History==0,'N','Y')
base <- ifelse(is.na(base), 'Y', base)
result_base <- data.frame(Loan_ID=load_test$Loan_ID, Loan_Status=base)

write.csv(result_gbm_first, file = 'D:/xxxx/result_gbm_first.csv', row.names = FALSE)
write.csv(result_base, file = 'D:/xxxx/result_base.csv', row.names = FALSE)
```

结果发现直接用Credit_History做判断结果比直接使用gbm模型效果好。

#### 缺失值处理以及类别变量处理

这里大部分变量都是二元变量，将空值置为缺失值，并用bagImpute进行填充。

```
factor_to_logistic <- function(train){
  
  train[,2:dim(train)[2]] <- lapply(train[,2:dim(train)[2]], as.character)
  train$Gender <- ifelse(train$Gender =='Male', 1, train$Gender)
  train$Gender <- ifelse(train$Gender =='Female', 0, train$Gender)
  train$Gender <- ifelse(train$Gender =='', NA, train$Gender)
  
  train$Married <- ifelse(train$Married =='Yes', 1, train$Married)
  train$Married <- ifelse(train$Married =='No', 0, train$Married)
  train$Married <- ifelse(train$Married =='', NA, train$Married)
  
  train$Education <- ifelse(train$Education == 'Graduate', 1, train$Education)
  train$Education <- ifelse(train$Education == 'Not Graduate', 0, train$Education)
  train$Education <- ifelse(train$Education =='', NA, train$Education)
  
  train$Self_Employed <- ifelse(train$Self_Employed == 'No', 1, train$Self_Employed)
  train$Self_Employed <- ifelse(train$Self_Employed == 'Yes', 0, train$Self_Employed)
  train$Self_Employed <- ifelse(train$Self_Employed =='', NA, train$Self_Employed)
  
  
  train$Dependents <- ifelse(train$Dependents=='', NA, train$Dependents)
  train$Dependents <- ifelse(train$Dependents=='3+', '3', train$Dependents)
  
  
  train <- dummy.data.frame(train, names = c("Property_Area"), sep = "_")
  
  train[,2:11] <- lapply(train[,2:11], as.numeric)
  
  return(train)
}
```

```
source('D:/xxxx/factor_to_logistic.R')
train_test <- rbind(load_train[,1:12], load_test)
train_test <- factor_to_logistic(train_test)
library('ipred')
preProcValues <- preProcess(train_test[,2:14], method = c('bagImpute'))
train_nona <- predict(preProcValues, train_test[,2:14])
train_test[,2:14] <- train_nona

train <- train_test[1:nrow(load_train), ]
test <- train_test[-(1:nrow(load_train)), ]
```

#### 添加聚类变量

由于数据量很小，这里采用系统聚类。

```
hclustmodel <- hclust(dist(train_test[,7:10]),method = "ward.D")
memb <- cutree(hclustmodel, k = 3)
table(memb)
train_test <- data.frame(train_test, memb=memb)
```

#### 重新建模

这里采用随机森林进行建模。

**首先不使用聚类变量**

```
rf_second <- train(train[,2:14], load_train[, 13], method = 'rf', trControl=ctrl,  metric = "ROC")
rf_second 
Random Forest 

614 samples
 13 predictor
  2 classes: 'N', 'Y' 

No pre-processing
Resampling: Cross-Validated (5 fold, repeated 1 times) 
Summary of sample sizes: 491, 491, 492, 491, 491 
Resampling results across tuning parameters:

  mtry  ROC        Sens       Spec     
   2    0.7603812  0.4426451  0.9645378
   7    0.7669420  0.4738192  0.9195518
  13    0.7607099  0.4581646  0.9053221

ROC was used to select the optimal model using  the largest value.
The final value used for the model was mtry = 7.
```

查看变量重要程度：

```
varImp(rf_second)
rf variable importance

                         Overall
Credit_History          100.0000
ApplicantIncome          57.7973
LoanAmount               52.6754
CoapplicantIncome        29.7756
Loan_Amount_Term          9.4909
Dependents                9.3701
Self_Employed             3.4527
Married                   2.7405
Gender                    1.9447
Property_Area_Semiurban   1.7957
Education                 1.7914
Property_Area_Rural       0.9429
Property_Area_Urban       0.0000
```

**加上聚类变量**

```
rf_second_hcl <- train(train[,2:15], load_train[, 13], method = 'rf', trControl=ctrl,  metric = "ROC")
rf_second_hcl
Random Forest 

614 samples
 14 predictor
  2 classes: 'N', 'Y' 

No pre-processing
Resampling: Cross-Validated (5 fold, repeated 1 times) 
Summary of sample sizes: 492, 492, 491, 491, 490 
Resampling results across tuning parameters:

  mtry  ROC        Sens       Spec     
   2    0.7588417  0.4319838  0.9621289
   8    0.7674908  0.4997301  0.9194678
  14    0.7610505  0.4839406  0.9123249

ROC was used to select the optimal model using  the largest value.
The final value used for the model was mtry = 8.
```

查看重要变量：

```
varImp(rf_second_hcl)
rf variable importance

                        Overall
Credit_History          100.000
ApplicantIncome          54.286
LoanAmount               49.242
CoapplicantIncome        28.206
Loan_Amount_Term          9.802
memb                      9.062
Dependents                9.043
Self_Employed             3.525
Married                   3.162
Property_Area_Semiurban   2.504
Education                 2.376
Gender                    1.914
Property_Area_Rural       1.871
Property_Area_Urban       0.000
```

可以看到这里的聚类变量memb比大部分变量都重要，并且模型整体的ROC也有一点提高。将以上模型应用于测试集，多次运行(test集太少，随机性很大)后目前最好的结果是0.805556。