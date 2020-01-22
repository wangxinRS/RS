# -*- coding:utf-8 -*-

from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np

#数据加载
data = pd.read_csv(r'F:\开课吧\RS基础课\RS6-master\L2\team_cluster\team_cluster_data.csv', encoding='gbk');
#print(data);
train_x = data[['2019国际排名', '2018世界杯排名', '2015亚洲杯排名']];
#print(train_x);
#规范到[0, 1]空间
min_max_scaler = preprocessing.MinMaxScaler();
train_x = min_max_scaler.fit_transform(train_x);
#print(train_x);

#kmeans算法
kmeans = KMeans(n_clusters=3);
kmeans.fit(train_x);
predict_y = kmeans.predict(train_x);
#print(predict_y);

#合并聚类
result = pd.concat((data, pd.DataFrame(predict_y)), axis=1);
result.rename({0: u'聚类结果'}, axis=1, inplace=True);
print(result);














