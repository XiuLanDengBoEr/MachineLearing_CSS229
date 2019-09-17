import numpy as np
import pandas as pd
from  matplotlib import pyplot as plt


train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
print(train.info())
print(test.info())
#选取对预测有效的特征
selected_features=["Pclass","Sex","Age","Embarked","SibSp","Parch","Fare"]
x_train=train[selected_features]
x_test=test[selected_features]
y_train=train["Survived"]


print(x_train["Embarked"].value_counts())
print(x_test["Embarked"].value_counts())

x_train["Embarked"].fillna('S',inplace=True)
x_test["Embarked"].fillna("S",inplace=True)
x_train["Age"].fillna(x_train["Age"].mean(),inplace=True)
x_test["Age"].fillna(x_test["Age"].mean(),inplace=True)
x_test["Fare"].fillna(x_test["Fare"].mean(),inplace=True)

print(x_train.info())
print(x_test.info())


from sklearn.feature_extraction import DictVectorizer
dic_vec=DictVectorizer(sparse=False)
#print(x_train.to_dict(orient='record'))
x_train=dic_vec.fit_transform(x_train.to_dict(orient="record"))
dic_vec.get_feature_names()

x_test=dic_vec.fit_transform(x_test.to_dict(orient="record"))

#导入集成学习的包RandomForest，开始使用初始化的参数
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()

#导入集成学习的包RandomForest，开始使用初始化的参数
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
#导入xgboost分类器，也是用初始配置
from xgboost import XGBClassifier
xgbc=XGBClassifier()

#导入交叉验证包，对默认参数的分类器进行打分
from sklearn.model_selection import cross_val_score
print(cross_val_score(rfc,x_train,y_train,cv=5).mean())
print("\n")
print(cross_val_score(xgbc,x_train,y_train,cv=5).mean())

#使用默认的RandomForestC进行分类训练并预测，将结果保存在rfc_submission.csv
rfc.fit(x_train,y_train)  #训练完毕
rfc_y_predict=rfc.predict(x_test)   #进行预测
#将结果写成df形式
rfc_submission=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':rfc_y_predict})
print(rfc_submission)
#保存结果
rfc_submission.to_csv("E:\\py_Coding\\rfc_submission2.csv",index=0,x)

#使用默认配置的XGBoost进行预测
xgbc.fit(x_train,y_train)
xgbc_y_predict=xgbc.predict(x_test)

xgbc_submission=pd.DataFrame({"PassengerId":test["PassengerId"],"Survived":xgbc_y_predict})
print(xgbc_submission)
xgbc_submission.to_csv("E:\\py_Coding\\xgbc_submission2.csv",index=0)

#使用并行网格搜索的方式寻找较好的超参数组合，优化xgbc的效果
from sklearn.model_selection import GridSearchCV
params={"max_depth":list(range(2,7)), "n_estimators":list(range(100,1100,200)), "learning_rate":[0.005,0.1,0.25,0.5,1]}
#构造梯度搜索器
xgbc_best=XGBClassifier()
xgb_grid_search=GridSearchCV(xgbc_best,params,n_jobs=-1,cv=5,verbose=1)
xgb_grid_search.fit(x_train,y_train)

print(xgb_grid_search.best_score_)
print(xgb_grid_search.best_params_)

xgbc_best_y_predict=xgb_grid_search.predict(x_test)
#correct_rate=len(xgbc_best_y_predict[xgbc_best_y_predict==y_test])/len(xgbc_best_y_predict)
#print(correct_rate)


xgbc_best_submission=pd.DataFrame({"PassengerId":test["PassengerId"],"Survived":xgbc_best_y_predict})
xgbc_best_submission.to_csv("E:\\py_Coding\\xgbc_best_submission1.csv",index=0)