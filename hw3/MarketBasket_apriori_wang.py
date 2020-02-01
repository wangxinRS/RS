# -*- coding:utf-8 -*-
# 分析MarketBasket中的频繁项集和关联规则
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# 数据加载
data = pd.read_csv(r'F:\开课吧\RS基础课\RS6-master\L3\MarketBasket\Market_Basket_Optimisation.csv', header=None);
# 数据探索
print(data.head());
print(data.info());
print(data.describe(include=['O']));
# 将每个transaction的item合并
def merge(line):
    res = line[0];
    for string in line[1:]:
        if pd.notnull(string):
            string = '|' + string;
            res = res + string;
    return res;

data = data.apply(merge, axis=1);
print(data.head())
# 将data进行one-hot编码
data_hot_encoded = data.str.get_dummies('|');
print(data_hot_encoded.head());

# 挖掘频繁项集，最小支持度为0.02
itemsets = apriori(data_hot_encoded, use_colnames=True, min_support=0.02);
# 将频繁项集按照支持度从大到小排序
itemsets = itemsets.sort_values(by='support', ascending=False);
print('-'*20, '频繁项集前五行', '-'*20)
print(itemsets.head());
print(len(itemsets));
# 根据频繁项集计算关联规则，设置最小提升度为2
rules = association_rules(itemsets, metric='lift', min_threshold=1.5);
# 按照提升度从大到小排序
rules = rules.sort_values(by='lift', ascending=False);
print('-'*20, '关联规则前五行', '-'*20);
print(rules.head());
print(len(rules));














