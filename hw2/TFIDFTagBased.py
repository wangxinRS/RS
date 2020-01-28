# -*- coding:utf-8 -*-
import math
import pandas as pd
import operator
import random

file_path = r'F:\开课吧\RS基础课\RS6-master\L2\delicious-2k\user_taggedbookmarks-timestamps.dat';

#生成records，根据records生成user_tags, user_items, tag_items
records = {};

train_data = {};
test_data = {};

user_tags = {};
user_items = {};
tag_items = {};

#对于标签t，有多少人使用
tag_users = {};

#载入数据并读取进records
def load():
    data = pd.read_csv(file_path, sep='\t');
    for i in range(len(data)):
        uid = data['userID'][i];
        iid = data['bookmarkID'][i];
        tid = data['tagID'][i];

        records.setdefault(uid, {});
        records[uid].setdefault(iid, []);
        records[uid][iid].append(tid);

load();

# 将records切分成train_data与test_data
def train_test_split(ratio, seed=1):
    random.seed(seed);
    for u, items in records.items():
        for item, tags in records[u].items():
            if random.random() < ratio:
                test_data.setdefault(u, {});
                test_data[u].setdefault(item, []);
                test_data[u][item].extend(tags);
            else:
                train_data.setdefault(u, {});
                train_data[u].setdefault(item, []);
                train_data[u][item].extend(tags);

train_test_split(0.2);

# 根据train_data生成user_items, user_tags, tag_items
def addValueToMat(mat, index, item, value=1):
        mat.setdefault(index, {});
        mat[index].setdefault(item, 0);
        mat[index][item] += value;

# 使用训练集，初始化user_items, user_tags, tag_items
def initStat():
    for u, items in train_data.items():
        for item, tags in train_data[u].items():
            for tag in tags:
                addValueToMat(user_tags, u, tag);
                addValueToMat(user_items, u, item);
                addValueToMat(tag_items, tag, item);
                addValueToMat(tag_users, tag, u);
    print('user_tags, user_items, tag_items初始化完成.');
    print('user_tags大小 %d, user_items大小 %d, tag_items大小 %d' % (len(user_tags), len(user_items), len(tag_items)));

initStat();


# 根据user_tags、user_items和tag_items来计算推荐
def recommend(user, N):
    #计算tag_users[t]，即对于标签t，有多少人使用
    count_tag_users = {};
    for tag, users in tag_users.items():
        count_tag_users.setdefault(tag, 0);
        for user, num in users.items():
            count_tag_users[tag] += num;

    recommend_items = {};
    tagged_items = user_items[user];
    for tag, wut in user_tags[user].items():
        for item, wti in tag_items[tag].items():
            if item in tagged_items:
                continue;
            recommend_items.setdefault(item, 0);
            recommend_items[item] += wut * wti / math.log(1+count_tag_users[tag], math.e);
    return sorted(recommend_items.items(), key=operator.itemgetter(1), reverse=True)[0: N];


#使用测试集，计算准确率和召回率
def precisionAndRecall(N):
    hit = 0;
    h_recall = 0;
    h_precision = 0;
    for user, items in test_data.items():
        if user not in train_data.keys():
            continue;
        rank = recommend(user, N);
        for item, rui in rank:
            if item in items:
                hit += 1;
        h_recall += len(items);
        h_precision += N;
    return (hit/(h_precision*1.0), hit/(h_recall*1.0));

print(precisionAndRecall(5));












