# file create: gen word_review_stemmed.csv
import time

start_time = time.time()
import pandas as pd
import re


def str_common_word(str1, str2):
    str2 = str2.lower().split(" ")
    if str1 in str2:
        cnt = 1
    else:
        cnt = 0
    return cnt


def str_common_word2(str1, str2):
    str2 = str(str2).lower()
    if str2.find(str1) >= 0:
        cnt = 1
    else:
        cnt = 0
    return cnt


df_all = pd.read_csv('df_all_rmse5_2.csv', encoding="ISO-8859-1")
df_all = df_all[['product_uid', 'search_term', 'product_title', 'product_description']]
df_all.reset_index(inplace=True)
df_attr = pd.read_csv('attributes_stemmed.csv', encoding="ISO-8859-1").fillna(" ")
print("--- Files Loaded: %s minutes ---" % round(((time.time() - start_time) / 60), 2))

d_prod_query = {}
for i in range(len(df_all)):
    b_ = str(df_all['product_uid'][i])
    if b_ not in d_prod_query:
        d_prod_query[b_] = [list(set(str(df_all['search_term'][i]).lower().split(" "))),
                            str(df_all['product_title'][i]).decode('utf-8'),
                            str(df_all['product_description'][i])]
    else:
        d_prod_query[b_][0] = list(
            set(d_prod_query[b_][0] + list(set(str(df_all['search_term'][i]).lower().split(" ")))))

f = open("dictionary_stemmed.txt", "w")
f.write(str(d_prod_query))
f.close()

print("--- Product & Search Term Dictionary: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
# stop_ = list(text.ENGLISH_STOP_WORDS)
stop_ = []
d = {}
for i in d_prod_query:
    a = d_prod_query[i][0]
    df_gen_attr = df_attr.loc[df_attr['product_uid'] == str(i) + ".0"]
    for b_ in a:
        if len(b_) > 0:
            col_lst = []
            for j in range(len(df_gen_attr)):
                if str_common_word(b_, df_gen_attr['value'].iloc[j]) > 0:
                    col_lst.append(df_gen_attr['name'].iloc[j])
            if b_ not in d:
                d[b_] = [1, str_common_word(b_, d_prod_query[i][1]), str_common_word2(b_, d_prod_query[i][1]),
                         col_lst[:]]
            else:
                d[b_][0] += 1
                d[b_][1] += str_common_word(b_, d_prod_query[i][1])
                d[b_][2] += str_common_word2(b_, d_prod_query[i][1])
                d[b_][3] = list(set(d[b_][3] + col_lst))

ds2 = pd.DataFrame.from_dict(d, orient='index')
ds2.columns = ['count', 'in title 1', 'in title 2', 'attribute type']
ds2 = ds2.sort_values(by=['count'], ascending=[False])

f = open("word_review_stemmed.csv", "w")
f.write("word|count|in title 1|in title 2|attribute type\n")
for i in range(len(ds2)):
    f.write(ds2.index[i] + "|" + str(ds2["count"][i]) + "|" + str(ds2["in title 1"][i]) + "|" + str(
            ds2["in title 2"][i]) + "|" + str(ds2["attribute type"][i]) + "\n")
f.close()
print("--- File Created: %s minutes ---" % round(((time.time() - start_time) / 60), 2))

# file create :gen
import time

start_time = time.time()

import numpy as np
import pandas as pd
from nltk.metrics import edit_distance
import re

df_all = pd.read_csv('df_all_rmse5_2.csv', encoding="ISO-8859-1", index_col=0)
df_temp = df_all[:]

# v1
dtest = pd.read_csv('word_review_stemmed.csv', encoding="ISO-8859-1", index_col=0, sep='|').to_dict('index')
dm_attr = [
    [z, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0,
     -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0] for z in range(len(df_temp))]
for a in range(len(df_temp)):
    b = [z for z in str(df_temp.search_term[a]).split(" ") if len(z) > 0]
    for c in range(1, len(b) + 1):
        d = str(b[c - 1]).lower()
        # d = re.sub(r"([0-9]),([0-9])", r"\1\2", d)
        if d in dtest:
            dm_attr[a][c - 1] = dtest[d]['in title 1'] / dtest[d]['count']
        else:
            dm_attr[a][c - 1] = 0.0
df_dm_attr = pd.DataFrame(dm_attr)
df_dm_attr.columns = ['sft01', 'sft02', 'sft03', 'sft04', 'sft05', 'sft06', 'sft07', 'sft08', 'sft09', 'sft10', 'sft11',
                      'sft12', 'sft13', 'sft14', 'sft101', 'sft102', 'sft103', 'sft104', 'sft105', 'sft106', 'sft107',
                      'sft108', 'sft109', 'sft110', 'sft111', 'sft112', 'sft113', 'sft114']
df_all = pd.concat([df_all, df_dm_attr], axis=1)
print("--- V1 Complete: %s minutes ---" % round(((time.time() - start_time) / 60), 2))

# v2
dm_attr = [
    [z, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0,
     -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0] for z in range(len(df_temp))]
for a in range(len(df_temp)):
    b = [z for z in str(df_temp.search_term[a]).split(" ") if len(z) > 0]
    for c in range(1, len(b) + 1):
        d = str(b[c - 1]).lower()
        # d = re.sub(r"([0-9]),([0-9])", r"\1\2", d)
        if d in dtest:
            dm_attr[a][c - 1] = dtest[d]['in title 2'] / dtest[d]['count']
        else:
            dm_attr[a][c - 1] = 0.0
df_dm_attr = pd.DataFrame(dm_attr)
df_dm_attr.columns = ['sftx01', 'sftx02', 'sftx03', 'sftx04', 'sftx05', 'sftx06', 'sftx07', 'sftx08', 'sftx09',
                      'sftx10', 'sftx11', 'sftx12', 'sftx13', 'sftx14', 'sftx101', 'sftx102', 'sftx103', 'sftx104',
                      'sftx105', 'sftx106', 'sftx107', 'sftx108', 'sftx109', 'sftx110', 'sftx111', 'sftx112', 'sftx113',
                      'sftx114']
df_all = pd.concat([df_all, df_dm_attr], axis=1)
print("--- V2 Complete: %s minutes ---" % round(((time.time() - start_time) / 60), 2))

# v3
[[z, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0,
  -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0] for z in range(len(df_temp))]
for a in range(len(df_temp)):
    b = [z for z in str(df_temp.search_term[a]).split(" ") if len(z) > 0]
    for c in range(1, len(b) + 1):
        d = str(b[c - 1]).lower()
        # d = re.sub(r"([0-9]),([0-9])", r"\1\2", d)
        if d in dtest:
            dm_attr[a][c - 1] = len(dtest[d]['attribute type'].split(","))
        else:
            dm_attr[a][c - 1] = 0.0
df_dm_attr = pd.DataFrame(dm_attr)
df_dm_attr.columns = ['sftz01', 'sftz02', 'sftz03', 'sftz04', 'sftz05', 'sftz06', 'sftz07', 'sftz08', 'sftz09',
                      'sftz10', 'sftz11', 'sftz12', 'sftz13', 'sftz14', 'sftz101', 'sftz102', 'sftz103', 'sftz104',
                      'sftz105', 'sftz106', 'sftz107', 'sftz108', 'sftz109', 'sftz110', 'sftz111', 'sftz112', 'sftz113',
                      'sftz114']
df_all = pd.concat([df_all, df_dm_attr], axis=1)
print("--- V3 Complete: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
df_all.to_csv('df_all_rmse5_3.csv', encoding="ISO-8859-1")
print("--- File Created: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
