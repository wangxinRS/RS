# -*- coding:utf-8 -*-

# 手肘法
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 输入数据
data = pd.read_csv(r'F:\开课吧\RS基础课\RS6-master\L3\team_cluster\team_cluster_data.csv', encoding='gbk');
# 规范化到[0, 1]空间
train_x = data.iloc[:, 1:];
min_max_scaler = preprocessing.MinMaxScaler();
train_x = min_max_scaler.fit_transform(train_x);
print(train_x);

# 统计不同K取值的误差平方和
sse = [];
for k in range(1, 11):
    # kmeans算法
    kmeans = KMeans(n_clusters=k);
    kmeans.fit(train_x);
    # 计算inertia簇内误差平方和
    sse.append(kmeans.inertia_);

x = range(1, 11);
plt.plot(x, sse, 'o-');
plt.xlabel('K');
plt.ylabel('SSE');
plt.show();

# 这里设置K=3
kmeans = KMeans(n_clusters=3);
kmeans.fit(train_x);
predict_y = kmeans.predict(train_x);
# 合并聚类结果，插入到原数据中
result = pd.concat([data, pd.DataFrame(predict_y, columns=['聚类结果'])], axis=1);




