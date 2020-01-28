# -*- coding:utf-8 -*-

import random
import math
import operator
import pandas as pd

file_path = r'F:\开课吧\RS基础课\RS6-master\L2\delicious-2k\user_taggedbookmarks-timestamps.dat';

data = pd.read_csv(file_path, sep='\t');

records = {};

# 训练集、测试集
train_data = {};
test_data = {};

# 用户标签、商品标签
user_tags = {};
user_items = {};
tag_items = {};

# 数据加载
def load():
    print('数据加载中...');
    data = pd.read_csv(file_path, sep='\t');
    for i in range(len(data)):
        uid = data['userID'][i];
        iid = data['bookmarkID'][i];
        tag = data['tagID'][i];
        records.setdefault(uid, {});
        records[uid].setdefault(iid, []);
        records[uid][iid].append(tag);

    print('数据集大小为 %d.' % len(data));
    print('设置tag的人数为 %d.' % len(records));
    print('数据加载完成!\n');


load();

# 将数据集拆分为训练集和测试集
def train_test_split(ratio, seed=100):
    random.seed(seed);
    for u in records.keys():
        for i in records[u].keys():
            if random.random() < ratio:
                test_data.setdefault(u, {});
                test_data[u].setdefault(i, []);
                test_data[u][i].extend(records[u][i]);
            else:
                train_data.setdefault(u, {});
                train_data[u].setdefault(i, []);
                train_data[u][i].extend(records[u][i]);
    print('训练集样本数 %d, 测试集样本数 %d \n' % (len(train_data), len(test_data)));

train_test_split(0.2, seed=100);

# 设置矩阵
def addValueToMat(mat, index, item, value=1):
    if index not in mat:
        mat.setdefault(index, {});
        mat[index].setdefault(item, value);
    else:
        if item not in mat[index]:
            mat[index].setdefault(item, value);
        else:
            mat[index][item] += value;

# 使用训练集，初始化user_tags, tag_items, user_items
def initStat():
    records = train_data;
    for u, items in records.items():
        for i, tags in items.items():
            for tag in tags:
                addValueToMat(user_tags, u, tag, 1);
                addValueToMat(user_items, u, i, 1);
                addValueToMat(tag_items, tag, i, 1);
    print('user_tags, tag_items, user_items 初始化完成.');
    print('user_tags大小 %d, tag_items大小 %d, user_items大小 %d \n' % (len(user_tags), len(tag_items), len(user_items)));

initStat();

# 对用户user推荐Top-N
def recommend(user, N):
    #计算sum(user_tags[u, t], t)和sum(tag_items[t, i], i)
    count_wut = 0;
    count_wti = {};
    for tag, wut in user_tags[user].items():
        count_wut += wut;

        count_wti[tag] = 0;
        for item, wti in tag_items[tag].items():
            count_wti[tag] += wti;

    #计算score(u, i)
    recommend_items = dict();
    tagged_items = user_items[user];
    for tag, wut in user_tags[user].items():
        for item, wti in tag_items[tag].items():
            if item in tagged_items:
                continue;
            recommend_items.setdefault(item, 0);
            recommend_items[item] += wut / count_wut * wti / count_wti[tag];

    return sorted(recommend_items.items(), key=operator.itemgetter(1), reverse=True)[0: N];



# 使用测试集，计算准确率和召回率
def precisionAndRecall(N):
    hit = 0;
    h_recall = 0;
    h_precision = 0;
    for user, items in test_data.items():
        if user not in train_data:
            continue;
        rank = recommend(user, N);
        for item, rui in rank:
            if item in items:
                hit += 1;
        h_recall += len(items);
        h_precision += N;
    return (hit / (h_precision * 1.0), hit / (h_recall * 1.0));

print(precisionAndRecall(5));






















