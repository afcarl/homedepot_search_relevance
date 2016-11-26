import time
start_time = time.time()

import pandas as pd


def str_common_word(str1, str2):
    str2 = str2.lower().split(" ")
    if str1 in str2:
        cnt=1
    else:
        cnt=0
    return cnt

def str_common_word2(str1, str2):
    str2 = str(str2).lower()
    if str2.find(str1)>=0:
        cnt=1
    else:
        cnt=0
    return cnt

df_train = pd.read_csv('../data/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('../data/test.csv', encoding="ISO-8859-1")
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = df_all[['product_uid','search_term','product_title']]
df_all.reset_index(inplace=True)

df_prod = pd.read_csv('../data/product_descriptions.csv', encoding='ISO-8859-1').fillna(" ")
df_attr = pd.read_csv('../data/attributes.csv', encoding='ISO-8859-1').fillna(" ")
print("--- Files Loaded: %s minutes ---" % round(((time.time() - start_time)/60),2))

d_prod_query = {}
for i in range(len(df_all)):
    b_ = str(df_all['product_uid'][i])
    if b_ not in d_prod_query:
        d_prod_query[b_] = [list(set(str(df_all['search_term'][i]).lower().split(" "))),
                            str(df_all['product_title'][i].encode('utf-8')).lower(),
                            str(df_prod.loc[df_prod['product_uid'] == df_all['product_uid'][i]]['product_description'].iloc[0]).lower()]
    else:
        d_prod_query[b_][0] = list(set(d_prod_query[b_][0] + list(set(str(df_all['search_term'][i]).lower().split(" ")))))

f = open("dictionary.txt", "w")
f.write(str(d_prod_query))
f.close()

print("--- Product & Search Term Dictionary: %s minutes ---" % round(((time.time() - start_time)/60),2))
#stop_ = list(text.ENGLISH_STOP_WORDS)
stop_ = []
d={}
for i in d_prod_query:
    a = d_prod_query[i][0]
    df_gen_attr = df_attr.loc[df_attr['product_uid'] == int(i)]
    for b_ in a:
        if len(b_)>0:
            col_lst = []
            for j in range(len(df_gen_attr)):
                if str_common_word(b_, df_gen_attr['value'].iloc[j])>0:
                    col_lst.append(df_gen_attr['name'].iloc[j])
            #if b_ not in stop_:
            if b_ not in d:
                d[b_] = [1,str_common_word(b_, d_prod_query[i][1]),str_common_word2(b_, d_prod_query[i][1]),col_lst[:]]
            else:
                d[b_][0] += 1
                d[b_][1] += str_common_word(b_, d_prod_query[i][1])
                d[b_][2] += str_common_word2(b_, d_prod_query[i][1])
                d[b_][3] =  list(set(d[b_][3] + col_lst))

ds2 = pd.DataFrame.from_dict(d,orient='index')
ds2.columns = ['count','in title 1','in title 2','attribute type']
ds2 = ds2.sort_values(by=['count'], ascending=[False])

f = open("word_review_v2.csv", "w")
f.write("word|count|in title 1|in title 2|attribute type\n")
for i in range(len(ds2)):
    f.write(ds2.index[i] + "|" + str(ds2["count"][i]) + "|" + str(ds2["in title 1"][i]) + "|" + str(ds2["in title 2"][i]) + "|" + str(ds2["attribute type"][i]) + "\n")
f.close()
print("--- File Created: %s minutes ---" % round(((time.time() - start_time)/60),2))

import time
start_time = time.time()

import numpy as np
import pandas as pd
from nltk.metrics import edit_distance
import re

df_train = pd.read_csv('../data/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('../data/test.csv', encoding="ISO-8859-1")
df_temp = pd.concat((df_train, df_test), axis=0, ignore_index=True)
num_train = df_train.shape[0]
df_all = pd.read_csv('df_all_rmse5.csv', encoding="ISO-8859-1", index_col=0)

#v1
dtest = pd.read_csv('word_review_v2.csv', encoding="ISO-8859-1", index_col=0, sep='|').to_dict('index')
dm_attr = [[z,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0] for z in range(len(df_temp))]
for a in range(len(df_temp)):
    b = [z for z in str(df_temp.search_term[a]).split(" ") if len(z)>0]
    for c in range(1,len(b)+1):
        d = str(b[c-1]).lower()
        d = re.sub(r"([0-9]),([0-9])", r"\1\2", d)
        if d in dtest:
            dm_attr[a][c-1] = dtest[d]['in title 1'] / dtest[d]['count']
        else:
            dm_attr[a][c-1] = 0.0
df_dm_attr = pd.DataFrame(dm_attr)
df_dm_attr.columns = ['ft01','ft02','ft03','ft04','ft05','ft06','ft07','ft08','ft09','ft10','ft11','ft12','ft13','ft14']
df_all = pd.concat([df_all, df_dm_attr], axis=1)
#v2
dm_attr = [[z,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0] for z in range(len(df_temp))]
for a in range(len(df_temp)):
    b = [z for z in str(df_temp.search_term[a]).split(" ") if len(z)>0]
    for c in range(1,len(b)+1):
        d = str(b[c-1]).lower()
        d = re.sub(r"([0-9]),([0-9])", r"\1\2", d)
        if d in dtest:
            dm_attr[a][c-1] = dtest[d]['in title 2'] / dtest[d]['count']
        else:
            dm_attr[a][c-1] = 0.0
df_dm_attr = pd.DataFrame(dm_attr)
df_dm_attr.columns = ['ftx01','ftx02','ftx03','ftx04','ftx05','ftx06','ftx07','ftx08','ftx09','ftx10','ftx11','ftx12','ftx13','ftx14']
df_all = pd.concat([df_all, df_dm_attr], axis=1)
#v3
dm_attr = [[z,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0,-99.0] for z in range(len(df_temp))]
for a in range(len(df_temp)):
    b = [z for z in str(df_temp.search_term[a]).split(" ") if len(z)>0]
    for c in range(1,len(b)+1):
        d = str(b[c-1]).lower()
        d = re.sub(r"([0-9]),([0-9])", r"\1\2", d)
        if d in dtest:
            dm_attr[a][c-1] = len(dtest[d]['attribute type'].split(","))
        else:
            dm_attr[a][c-1] = 0.0
df_dm_attr = pd.DataFrame(dm_attr)
df_dm_attr.columns = ['ftz01','ftz02','ftz03','ftz04','ftz05','ftz06','ftz07','ftz08','ftz09','ftz10','ftz11','ftz12','ftz13','ftz14']
df_all = pd.concat([df_all, df_dm_attr], axis=1)

df_all.to_csv('df_all_rmse5_2.csv')
print("--- File Created: %s minutes ---" % round(((time.time() - start_time)/60),2))