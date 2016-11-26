# -*- coding: ISO-8859-1 -*-
'''
Coding Just for Fun
Created by burness on 16/3/3.
'''
# import time
#
# start_time = time.time()
#
# import numpy as np
# import pandas as pd
# import sys
#
# reload(sys)
# sys.setdefaultencoding("utf-8")
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.ensemble import RandomForestRegressor
# # from sklearn import pipeline, model_selection
# from sklearn import pipeline, grid_search
# # from sklearn.feature_extraction import DictVectorizer
# from sklearn.pipeline import FeatureUnion
# from sklearn.decomposition import TruncatedSVD
# # from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import mean_squared_error, make_scorer
# # from nltk.metrics import edit_distance
# from nltk.stem.porter import *
#
# stemmer = PorterStemmer()
# # from nltk.stem.snowball import SnowballStemmer #0.003 improvement but takes twice as long as PorterStemmer
# # stemmer = SnowballStemmer('english')
# import re
# # import enchant
# import random
# import unicodedata
#
# from scipy.stats import randint as sp_randint
#
# random.seed(2016)
#
# df_train = pd.read_csv('../data/train.csv', encoding="ISO-8859-1")
# df_test = pd.read_csv('../data/test.csv', encoding="ISO-8859-1")
# df_pro_desc = pd.read_csv('../data/product_descriptions.csv', encoding="ISO-8859-1")
# df_attr = pd.read_csv('../data/attributes.csv', encoding="ISO-8859-1")
# df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
# # print df_brand.head()
# num_train = df_train.shape[0]
# df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
# # print df_all.head()
# df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
# # print df_all.head()
# df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
#
# print("--- Files Loaded: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
#
# # stop_w = ['for', 'xbi', 'and', 'in', 'th','on','sku','with','what','from','that','less','er','ing'] #'electr','paint','pipe','light','kitchen','wood','outdoor','door','bathroom'
# strNum = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 0}
# f = open('../data/spelling.txt', 'r')
# zspell = {}
# for line in f:
#     a, b = line.strip("\n").split("|")
#     zspell[a] = b
# f.close()
#
#
# def str_stem(s):
#     s = unicodedata.normalize('NFD', unicode(s)).encode('ascii', 'ignore')
#     s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s)  # Split words with a.A
#     s = s.lower()
#     s = s.replace("  ", " ")
#     s = re.sub(r"([0-9])( *),( *)([0-9])", r"\1\4", s)
#     s = s.replace(",", " ")
#     s = s.replace("$", " ")
#     s = s.replace("?", " ")
#     s = s.replace("-", " ")
#     s = s.replace("//", "/")
#     s = s.replace("..", ".")
#     s = s.replace(" / ", " ")
#     s = s.replace(" \\ ", " ")
#     s = s.replace(".", " . ")
#     s = s.replace("   ", " ")
#     s = s.replace("  ", " ").strip(" ")
#     s = re.sub(r"(.*)\.$", r"\1", s)  # end period
#     s = re.sub(r"(.*)\/$", r"\1", s)  # end period
#     s = re.sub(r"^\.(.*)", r"\1", s)  # start period
#     s = re.sub(r"^\/(.*)", r"\1", s)  # start slash
#     s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
#     s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
#     s = s.replace(" x ", " xbi ")
#     s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
#     s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
#     s = s.replace("*", " xbi ")
#     s = s.replace(" by ", " xbi ")
#     s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
#     s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
#     s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
#     s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
#     s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
#     s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
#     s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
#     s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
#     s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
#     s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
#     s = s.replace("°", " degrees ")
#     s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
#     s = s.replace(" v ", " volts ")
#     s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
#     s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
#     s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
#     s = s.replace("  ", " ")
#     s = s.replace(" . ", " ")
#     # s = (" ").join([z for z in s.split(" ") if z not in stop_w])
#     s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
#     s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
#     s = s.lower()
#     s = (" ").join([str(zspell[z]) if z in zspell else z for z in s.split(" ")])
#     return s
#
#
# def seg_words(str1, str2):
#     str2 = str2.lower()
#     str2 = re.sub("[^a-z0-9./]", " ", str2)
#     str2 = [z for z in set(str2.split()) if len(z) > 2]
#     words = str1.lower().split(" ")
#     s = []
#     for word in words:
#         if len(word) > 3:
#             s1 = []
#             s1 += segmentit(word, str2, True)
#             if len(s) > 1:
#                 s += [z for z in s1 if z not in ['er', 'ing', 's', 'less'] and len(z) > 1]
#             else:
#                 s.append(word)
#         else:
#             s.append(word)
#     return (" ".join(s))
#
#
# def segmentit(s, txt_arr, t):
#     st = s
#     r = []
#     for j in range(len(s)):
#         for word in txt_arr:
#             if word == s[:-j]:
#                 r.append(s[:-j])
#                 # print(s[:-j],s[len(s)-j:])
#                 s = s[len(s) - j:]
#                 r += segmentit(s, txt_arr, False)
#     if t:
#         i = len(("").join(r))
#         if not i == len(st):
#             r.append(st[i:])
#     return r
#
#
# def str_common_word(str1, str2):
#     words, cnt = str1.split(), 0
#     for word in words:
#         if str2.find(word) >= 0:
#             cnt += 1
#     return cnt
#
#
# def str_whole_word(str1, str2, i_):
#     cnt = 0
#     while i_ < len(str2):
#         i_ = str2.find(str1, i_)
#         if i_ == -1:
#             return cnt
#         else:
#             cnt += 1
#             i_ += len(str1)
#     return cnt
#
#
# # comment out the lines below use df_all.csv for further grid search testing
# # if adding features consider any drops on the 'cust_regression_vals' class
# # *** would be nice to have a file reuse option or script chaining option on Kaggle Scripts ***
# df_all['search_term'] = df_all['search_term'].map(lambda x: str_stem(x))
# df_all['product_title'] = df_all['product_title'].map(lambda x: str_stem(x))
# df_all['product_description'] = df_all['product_description'].map(lambda x: str_stem(x))
# df_all['brand'] = df_all['brand'].map(lambda x: str_stem(x))
# print("--- Stemming: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
# df_all['product_info'] = df_all['search_term'] + "\t" + df_all['product_title'] + "\t" + df_all['product_description']
# print("--- Prod Info: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
# df_all['len_of_query'] = df_all['search_term'].map(lambda x: len(x.split())).astype(np.int64)
# df_all['len_of_title'] = df_all['product_title'].map(lambda x: len(x.split())).astype(np.int64)
# df_all['len_of_description'] = df_all['product_description'].map(lambda x: len(x.split())).astype(np.int64)
# df_all['len_of_brand'] = df_all['brand'].map(lambda x: len(x.split())).astype(np.int64)
# print("--- Len of: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
# df_all['search_term'] = df_all['product_info'].map(lambda x: seg_words(x.split('\t')[0], x.split('\t')[1]))
# # print("--- Search Term Segment: %s minutes ---" % round(((time.time() - start_time)/60),2))
# df_all['query_in_title'] = df_all['product_info'].map(lambda x: str_whole_word(x.split('\t')[0], x.split('\t')[1], 0))
# df_all['query_in_description'] = df_all['product_info'].map(
#     lambda x: str_whole_word(x.split('\t')[0], x.split('\t')[2], 0))
# print("--- Query In: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
# df_all['query_last_word_in_title'] = df_all['product_info'].map(
#     lambda x: str_common_word(x.split('\t')[0].split(" ")[-1], x.split('\t')[1]))
# df_all['query_last_word_in_description'] = df_all['product_info'].map(
#     lambda x: str_common_word(x.split('\t')[0].split(" ")[-1], x.split('\t')[2]))
# print("--- Query Last Word In: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
# df_all['word_in_title'] = df_all['product_info'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
# df_all['word_in_description'] = df_all['product_info'].map(
#     lambda x: str_common_word(x.split('\t')[0], x.split('\t')[2]))
# df_all['ratio_title'] = df_all['word_in_title'] / df_all['len_of_query']
# df_all['ratio_description'] = df_all['word_in_description'] / df_all['len_of_query']
# df_all['attr'] = df_all['search_term'] + "\t" + df_all['brand']
# df_all['word_in_brand'] = df_all['attr'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
# df_all['ratio_brand'] = df_all['word_in_brand'] / df_all['len_of_brand']
# df_brand = pd.unique(df_all.brand.ravel())
# d = {}
# i = 1000
# for s in df_brand:
#     d[s] = i
#     i += 3
# df_all['brand_feature'] = df_all['brand'].map(lambda x: d[x])
# df_all['search_term_feature'] = df_all['search_term'].map(lambda x: len(x))
# df_all.to_csv('df_all_rmse5.csv', encoding="ISO-8859-1")
# df_all = pd.read_csv('df_all_rmse5.csv', encoding="ISO-8859-1", index_col=0)
# df_train = df_all.iloc[:num_train]
# df_test = df_all.iloc[num_train:]
# id_test = df_test['id']
# y_train = df_train['relevance'].values
# X_train = df_train[:]
# X_test = df_test[:]
# print("--- Features Set: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
#
# # Feature Creation
# import time
#
# start_time = time.time()
#
# import numpy as np
# import pandas as pd
# from nltk.metrics import edit_distance
# import re
#
#
# def str_common_word(str1, str2):
#     str2 = str2.lower().split(" ")
#     if str1 in str2:
#         cnt = 1
#     else:
#         cnt = 0
#     return cnt
#
#
# def str_common_word2(str1, str2):
#     str2 = str(str2).lower()
#     if str2.find(str1) >= 0:
#         cnt = 1
#     else:
#         cnt = 0
#     return cnt
#
#
# df_train = pd.read_csv('../data/train.csv', encoding="ISO-8859-1")
# df_test = pd.read_csv('../data/test.csv', encoding="ISO-8859-1")
# df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
# df_all = df_all[['product_uid', 'search_term', 'product_title']]
# df_all.reset_index(inplace=True)
#
# df_prod = pd.read_csv('../data/product_descriptions.csv', encoding="ISO-8859-1").fillna(" ")
# df_attr = pd.read_csv('../data/attributes.csv', encoding="ISO-8859-1").fillna(" ")
# print("--- Files Loaded: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
#
# d_prod_query = {}
# for i in range(len(df_all)):
#     b_ = str(df_all['product_uid'][i])
#     if b_ not in d_prod_query:
#         d_prod_query[b_] = [list(set(str(df_all['search_term'][i]).lower().split(" "))),
#                             str(df_all['product_title'][i].encode('utf-8')).lower(),
#                             str(df_prod.loc[df_prod['product_uid'] == df_all['product_uid'][i]][
#                                     'product_description'].iloc[0]).lower()]
#     else:
#         d_prod_query[b_][0] = list(
#             set(d_prod_query[b_][0] + list(set(str(df_all['search_term'][i]).lower().split(" ")))))
#
# f = open("../data/dictionary.txt", "w")
# f.write(str(d_prod_query))
# f.close()
#
# print("--- Product & Search Term Dictionary: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
# # stop_ = list(text.ENGLISH_STOP_WORDS)
# stop_ = []
# d = {}
# for i in d_prod_query:
#     a = d_prod_query[i][0]
#     df_gen_attr = df_attr.loc[df_attr['product_uid'] == int(i)]
#     for b_ in a:
#         if len(b_) > 0:
#             col_lst = []
#             for j in range(len(df_gen_attr)):
#                 if str_common_word(b_, df_gen_attr['value'].iloc[j]) > 0:
#                     col_lst.append(df_gen_attr['name'].iloc[j])
#             # if b_ not in stop_:
#             if b_ not in d:
#                 d[b_] = [1, str_common_word(b_, d_prod_query[i][1]), str_common_word2(b_, d_prod_query[i][1]),
#                          col_lst[:]]
#             else:
#                 d[b_][0] += 1
#                 d[b_][1] += str_common_word(b_, d_prod_query[i][1])
#                 d[b_][2] += str_common_word2(b_, d_prod_query[i][1])
#                 d[b_][3] = list(set(d[b_][3] + col_lst))
#
# ds2 = pd.DataFrame.from_dict(d, orient='index')
# ds2.columns = ['count', 'in title 1', 'in title 2', 'attribute type']
# ds2 = ds2.sort_values(by=['count'], ascending=[False])
#
# f = open("word_review_v2.csv", "w")
# f.write("word|count|in title 1|in title 2|attribute type\n")
# for i in range(len(ds2)):
#     f.write(ds2.index[i] + "|" + str(ds2["count"][i]) + "|" + str(ds2["in title 1"][i]) + "|" + str(
#             ds2["in title 2"][i]) + "|" + str(ds2["attribute type"][i]) + "\n")
# f.close()
# print("--- File Created: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
#
# # File create : gen df_all_rmse5_2.csv
# import time
#
# start_time = time.time()
#
# import numpy as np
# import pandas as pd
# from nltk.metrics import edit_distance
# import re
#
# df_train = pd.read_csv('../data/train.csv', encoding="ISO-8859-1")
# df_test = pd.read_csv('../data/test.csv', encoding="ISO-8859-1")
# df_temp = pd.concat((df_train, df_test), axis=0, ignore_index=True)
# num_train = df_train.shape[0]
# df_all = pd.read_csv('df_all.csv', encoding="ISO-8859-1", index_col=0)
#
# # v1
# dtest = pd.read_csv('word_review_v2.csv', encoding="ISO-8859-1", index_col=0, sep='|').to_dict('index')
# dm_attr = [[z, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0] for z in
#            range(len(df_temp))]
# for a in range(len(df_temp)):
#     b = [z for z in str(df_temp.search_term[a]).split(" ") if len(z) > 0]
#     for c in range(1, len(b) + 1):
#         d = str(b[c - 1]).lower()
#         d = re.sub(r"([0-9]),([0-9])", r"\1\2", d)
#         if d in dtest:
#             dm_attr[a][c - 1] = dtest[d]['in title 1'] / dtest[d]['count']
#         else:
#             dm_attr[a][c - 1] = 0.0
# df_dm_attr = pd.DataFrame(dm_attr)
# df_dm_attr.columns = ['ft01', 'ft02', 'ft03', 'ft04', 'ft05', 'ft06', 'ft07', 'ft08', 'ft09', 'ft10', 'ft11', 'ft12',
#                       'ft13', 'ft14']
# df_all = pd.concat([df_all, df_dm_attr], axis=1)
# # v2
# dm_attr = [[z, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0] for z in
#            range(len(df_temp))]
# for a in range(len(df_temp)):
#     b = [z for z in str(df_temp.search_term[a]).split(" ") if len(z) > 0]
#     for c in range(1, len(b) + 1):
#         d = str(b[c - 1]).lower()
#         d = re.sub(r"([0-9]),([0-9])", r"\1\2", d)
#         if d in dtest:
#             dm_attr[a][c - 1] = dtest[d]['in title 2'] / dtest[d]['count']
#         else:
#             dm_attr[a][c - 1] = 0.0
# df_dm_attr = pd.DataFrame(dm_attr)
# df_dm_attr.columns = ['ftx01', 'ftx02', 'ftx03', 'ftx04', 'ftx05', 'ftx06', 'ftx07', 'ftx08', 'ftx09', 'ftx10', 'ftx11',
#                       'ftx12', 'ftx13', 'ftx14']
# df_all = pd.concat([df_all, df_dm_attr], axis=1)
# # v3
# dm_attr = [[z, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0] for z in
#            range(len(df_temp))]
# for a in range(len(df_temp)):
#     b = [z for z in str(df_temp.search_term[a]).split(" ") if len(z) > 0]
#     for c in range(1, len(b) + 1):
#         d = str(b[c - 1]).lower()
#         d = re.sub(r"([0-9]),([0-9])", r"\1\2", d)
#         if d in dtest:
#             dm_attr[a][c - 1] = len(dtest[d]['attribute type'].split(","))
#         else:
#             dm_attr[a][c - 1] = 0.0
# df_dm_attr = pd.DataFrame(dm_attr)
# df_dm_attr.columns = ['ftz01', 'ftz02', 'ftz03', 'ftz04', 'ftz05', 'ftz06', 'ftz07', 'ftz08', 'ftz09', 'ftz10', 'ftz11',
#                       'ftz12', 'ftz13', 'ftz14']
# df_all = pd.concat([df_all, df_dm_attr], axis=1)
#
# df_all.to_csv('df_all_rmse5_2.csv', encoding="ISO-8859-1")
# print("--- File Created: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
#
# # file create : gen attributes_semmed.csv
# import time
#
# start_time = time.time()
#
# import numpy as np
# import pandas as pd
# from nltk.metrics import edit_distance
# from nltk.stem.porter import *
#
# stemmer = PorterStemmer()
# import re
#
#
# def str_stem(s):
#     # if isinstance(s, str):
#     s = unicodedata.normalize('NFD', unicode(s)).encode('ascii', 'ignore')
#     s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s)  # Split words with a.A
#     s = s.lower()
#     s = s.replace("  ", " ")
#     s = re.sub(r"([0-9])( *),( *)([0-9])", r"\1\4", s)
#     s = s.replace(",", " ")
#     s = s.replace("$", " ")
#     s = s.replace("?", " ")
#     s = s.replace("-", " ")
#     s = s.replace("//", "/")
#     s = s.replace("..", ".")
#     s = s.replace(" / ", " ")
#     s = s.replace(" \\ ", " ")
#     s = s.replace(".", " . ")
#     s = s.replace("   ", " ")
#     s = s.replace("  ", " ").strip(" ")
#     s = re.sub(r"(.*)\.$", r"\1", s)  # end period
#     s = re.sub(r"(.*)\/$", r"\1", s)  # end slash
#     s = re.sub(r"^\.(.*)", r"\1", s)  # start period
#     s = re.sub(r"^\/(.*)", r"\1", s)  # start slash
#     s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
#     s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
#     s = s.replace(" x ", " xbi ")
#     s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
#     s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
#     s = s.replace("*", " xbi ")
#     s = s.replace(" by ", " xbi ")
#     s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
#     s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
#     s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
#     s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
#     s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
#     s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
#     s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
#     s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
#     s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
#     s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
#     s = s.replace("°", " degrees ")
#     s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
#     s = s.replace(" v ", " volts ")
#     s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
#     s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
#     s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
#     s = s.replace("  ", " ")
#     s = s.replace(" . ", " ")
#     s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
#     s = s.lower()
#     return s
#     # else:
#     #     return "null"
#
#
# df_attr = pd.read_csv('../data/attributes.csv').fillna(" ")
# print("--- Files Loaded: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
#
# df_attr['value'] = df_attr['value'].map(lambda x: str_stem(str(x)))
# df_attr.to_csv('attributes_stemmed.csv')
# print("--- File Created: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
#
# # file create: gen word_review_stemmed.csv
# import time
#
# start_time = time.time()
# import pandas as pd
# import re
#
#
# def str_common_word(str1, str2):
#     str2 = str2.lower().split(" ")
#     if str1 in str2:
#         cnt = 1
#     else:
#         cnt = 0
#     return cnt
#
#
# def str_common_word2(str1, str2):
#     str2 = str(str2).lower()
#     if str2.find(str1) >= 0:
#         cnt = 1
#     else:
#         cnt = 0
#     return cnt
#
#
# df_all = pd.read_csv('df_all_rmse5_2.csv', encoding="ISO-8859-1")
# df_all = df_all[['product_uid', 'search_term', 'product_title', 'product_description']]
# df_all.reset_index(inplace=True)
# df_attr = pd.read_csv('attributes_stemmed.csv', encoding="ISO-8859-1").fillna(" ")
# print("--- Files Loaded: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
#
# d_prod_query = {}
# for i in range(len(df_all)):
#     b_ = str(df_all['product_uid'][i])
#     if b_ not in d_prod_query:
#         d_prod_query[b_] = [list(set(str(df_all['search_term'][i]).lower().split(" "))),
#                             str(df_all['product_title'][i]).decode('utf-8'),
#                             str(df_all['product_description'][i])]
#     else:
#         d_prod_query[b_][0] = list(
#             set(d_prod_query[b_][0] + list(set(str(df_all['search_term'][i]).lower().split(" ")))))
#
# f = open("dictionary_stemmed.txt", "w")
# f.write(str(d_prod_query))
# f.close()
#
# print("--- Product & Search Term Dictionary: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
# # stop_ = list(text.ENGLISH_STOP_WORDS)
# stop_ = []
# d = {}
# for i in d_prod_query:
#     a = d_prod_query[i][0]
#     df_gen_attr = df_attr.loc[df_attr['product_uid'] == str(i) + ".0"]
#     for b_ in a:
#         if len(b_) > 0:
#             col_lst = []
#             for j in range(len(df_gen_attr)):
#                 if str_common_word(b_, df_gen_attr['value'].iloc[j]) > 0:
#                     col_lst.append(df_gen_attr['name'].iloc[j])
#             if b_ not in d:
#                 d[b_] = [1, str_common_word(b_, d_prod_query[i][1]), str_common_word2(b_, d_prod_query[i][1]),
#                          col_lst[:]]
#             else:
#                 d[b_][0] += 1
#                 d[b_][1] += str_common_word(b_, d_prod_query[i][1])
#                 d[b_][2] += str_common_word2(b_, d_prod_query[i][1])
#                 d[b_][3] = list(set(d[b_][3] + col_lst))
#
# ds2 = pd.DataFrame.from_dict(d, orient='index')
# ds2.columns = ['count', 'in title 1', 'in title 2', 'attribute type']
# ds2 = ds2.sort_values(by=['count'], ascending=[False])
#
# f = open("word_review_stemmed.csv", "w")
# f.write("word|count|in title 1|in title 2|attribute type\n")
# for i in range(len(ds2)):
#     f.write(ds2.index[i] + "|" + str(ds2["count"][i]) + "|" + str(ds2["in title 1"][i]) + "|" + str(
#             ds2["in title 2"][i]) + "|" + str(ds2["attribute type"][i]) + "\n")
# f.close()
# print("--- File Created: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
#
# # file create :gen
# import time
#
# start_time = time.time()
#
# import numpy as np
# import pandas as pd
# from nltk.metrics import edit_distance
# import re
#
# df_all = pd.read_csv('df_all_rmse5_2.csv', encoding="ISO-8859-1", index_col=0)
# df_temp = df_all[:]
#
# # v1
# dtest = pd.read_csv('word_review_stemmed.csv', encoding="ISO-8859-1", index_col=0, sep='|').to_dict('index')
# dm_attr = [
#     [z, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0,
#      -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0] for z in range(len(df_temp))]
# for a in range(len(df_temp)):
#     b = [z for z in str(df_temp.search_term[a]).split(" ") if len(z) > 0]
#     for c in range(1, len(b) + 1):
#         d = str(b[c - 1]).lower()
#         # d = re.sub(r"([0-9]),([0-9])", r"\1\2", d)
#         if d in dtest:
#             dm_attr[a][c - 1] = dtest[d]['in title 1'] / dtest[d]['count']
#         else:
#             dm_attr[a][c - 1] = 0.0
# df_dm_attr = pd.DataFrame(dm_attr)
# df_dm_attr.columns = ['sft01', 'sft02', 'sft03', 'sft04', 'sft05', 'sft06', 'sft07', 'sft08', 'sft09', 'sft10', 'sft11',
#                       'sft12', 'sft13', 'sft14', 'sft101', 'sft102', 'sft103', 'sft104', 'sft105', 'sft106', 'sft107',
#                       'sft108', 'sft109', 'sft110', 'sft111', 'sft112', 'sft113', 'sft114']
# df_all = pd.concat([df_all, df_dm_attr], axis=1)
# print("--- V1 Complete: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
#
# # v2
# dm_attr = [
#     [z, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0,
#      -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0] for z in range(len(df_temp))]
# for a in range(len(df_temp)):
#     b = [z for z in str(df_temp.search_term[a]).split(" ") if len(z) > 0]
#     for c in range(1, len(b) + 1):
#         d = str(b[c - 1]).lower()
#         # d = re.sub(r"([0-9]),([0-9])", r"\1\2", d)
#         if d in dtest:
#             dm_attr[a][c - 1] = dtest[d]['in title 2'] / dtest[d]['count']
#         else:
#             dm_attr[a][c - 1] = 0.0
# df_dm_attr = pd.DataFrame(dm_attr)
# df_dm_attr.columns = ['sftx01', 'sftx02', 'sftx03', 'sftx04', 'sftx05', 'sftx06', 'sftx07', 'sftx08', 'sftx09',
#                       'sftx10', 'sftx11', 'sftx12', 'sftx13', 'sftx14', 'sftx101', 'sftx102', 'sftx103', 'sftx104',
#                       'sftx105', 'sftx106', 'sftx107', 'sftx108', 'sftx109', 'sftx110', 'sftx111', 'sftx112', 'sftx113',
#                       'sftx114']
# df_all = pd.concat([df_all, df_dm_attr], axis=1)
# print("--- V2 Complete: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
#
# # v3
# [[z, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0,
#   -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0] for z in range(len(df_temp))]
# for a in range(len(df_temp)):
#     b = [z for z in str(df_temp.search_term[a]).split(" ") if len(z) > 0]
#     for c in range(1, len(b) + 1):
#         d = str(b[c - 1]).lower()
#         # d = re.sub(r"([0-9]),([0-9])", r"\1\2", d)
#         if d in dtest:
#             dm_attr[a][c - 1] = len(dtest[d]['attribute type'].split(","))
#         else:
#             dm_attr[a][c - 1] = 0.0
# df_dm_attr = pd.DataFrame(dm_attr)
# df_dm_attr.columns = ['sftz01', 'sftz02', 'sftz03', 'sftz04', 'sftz05', 'sftz06', 'sftz07', 'sftz08', 'sftz09',
#                       'sftz10', 'sftz11', 'sftz12', 'sftz13', 'sftz14', 'sftz101', 'sftz102', 'sftz103', 'sftz104',
#                       'sftz105', 'sftz106', 'sftz107', 'sftz108', 'sftz109', 'sftz110', 'sftz111', 'sftz112', 'sftz113',
#                       'sftz114']
# df_all = pd.concat([df_all, df_dm_attr], axis=1)
# print("--- V3 Complete: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
# df_all.to_csv('df_all_rmse5_3.csv', encoding="ISO-8859-1")
# print("--- File Created: %s minutes ---" % round(((time.time() - start_time) / 60), 2))

# file created:
# import time
#
# start_time = time.time()
# import pandas as pd
# import re
#
#
# def str_join_words(str1, str2):
#     s = (" ").join(["q_" + z for z in str1.split(" ")]) + " " + str2
#     return s
#
#
# df_all = pd.read_csv('df_all_rmse5_3.csv', low_memory=False,index_col=0, encoding="ISO-8859-1")
# import re
# import pandas
# import warnings
#
# myfile = 'df_all_rmse5_3.csv'
# target_type = str  # The desired output type
#
# with warnings.catch_warnings(record=True) as ws:
#     warnings.simplefilter("always")
#
#     df_all = pandas.read_csv(myfile, sep="|", header=None)
#     print("Warnings raised:", ws)
#     # We have an error on specific columns, try and load them as string
#     for w in ws:
#         s = str(w.message)
#         print("Warning message:", s)
#         match = re.search(r"Columns \(([0-9,]+)\) have mixed types\.", s)
#         if match:
#             columns = match.group(1).split(',') # Get columns as a list
#             columns = [int(c) for c in columns]
#             print("Applying %s dtype to columns:" % target_type, columns)
#             df_all.iloc[:,columns] = df_all.iloc[:,columns].astype(target_type)
# print("--- Files Loaded: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
# print df_all['product_info'].dtype
# df_all['search_and_prod_info'] = df_all['product_info'].map(
#     lambda x: str_join_words(x.split('\t')[0], x.split('\t')[1]), na_action='ignore')
# # df_all.fillna(-9999, inplace=True)
# print("--- Feature Created: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
# df_all.to_csv('df_all_rmse5_4.csv', encoding="ISO-8859-1")
# print("--- File Created: %s minutes ---" % round(((time.time() - start_time) / 60), 2))

import time

start_time = time.time()

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import pipeline  # model_selection
from sklearn.grid_search import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
from nltk.metrics import edit_distance
from nltk.stem.porter import *

stemmer = PorterStemmer()
import re
import random

random.seed(2016)
import xgboost as xgb
import sqlite3

df_train = pd.read_csv('../data/train.csv', encoding="ISO-8859-1")
num_train = df_train.shape[0]
df_all = pd.read_csv('df_all_rmse5_4.csv', encoding="ISO-8859-1", index_col=0)
train = df_all.iloc[:num_train]
test = df_all.iloc[num_train:]
print test.head()
id_test = test['id']
print id_test.head()
# balance train
c = sqlite3.connect(':memory:')
# c = sqlite3.connect('temp.db')
train.to_sql('t', c)
train = pd.read_sql('select * from t order by relevance desc, product_uid asc', c, index_col=['index'])
df_even = train.iloc[::2]  # even
df_odd = train.iloc[1::2]  # odd
train = pd.concat((df_even, df_odd), axis=0, ignore_index=True)
train = train.reset_index(drop=True)
y_train = train['relevance']
print("--- Features Set: %s minutes ---" % round(((time.time() - start_time) / 60), 2))


def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions) ** 0.5
    return fmean_squared_error_


RMSE = make_scorer(fmean_squared_error, greater_is_better=False)


class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, hd_searches):
        # d_col_drops=['id','relevance','search_term','product_title','product_description','product_info','attr','brand','search_and_prod_info','len_of_query', 'len_of_brand', 'query_in_description', 'word_in_brand', 'ratio_brand', 'ft04', 'ft05', 'ftx05', 'ftz04', 'ftz05', 'sft05', 'sftx05', 'sftz05','ft06', 'ft07', 'ft08', 'ft09', 'ft10', 'ft11', 'ft12', 'ft13', 'ft14', 'ftx06', 'ftx07', 'ftx08', 'ftx09', 'ftx10', 'ftx11', 'ftx12', 'ftx13', 'ftx14', 'ftz06', 'ftz07', 'ftz08', 'ftz09', 'ftz10', 'ftz11', 'ftz12', 'ftz13', 'ftz14', 'sft06', 'sft07', 'sft08', 'sft09', 'sft10', 'sft11', 'sft12', 'sft13', 'sft14', 'sft101', 'sft102', 'sft103', 'sft104', 'sft105', 'sft106', 'sft107', 'sft108', 'sft109', 'sft110', 'sft111', 'sft112', 'sft113', 'sft114', 'sftx06', 'sftx07', 'sftx08', 'sftx09', 'sftx10', 'sftx11', 'sftx12', 'sftx13', 'sftx14', 'sftx101', 'sftx102', 'sftx103', 'sftx104', 'sftx105', 'sftx106', 'sftx107', 'sftx108', 'sftx109', 'sftx110', 'sftx111', 'sftx112', 'sftx113', 'sftx114', 'sftz06', 'sftz07', 'sftz08', 'sftz09', 'sftz10', 'sftz11', 'sftz12', 'sftz13', 'sftz14', 'sftz101', 'sftz102', 'sftz103', 'sftz104', 'sftz105', 'sftz106', 'sftz107', 'sftz108', 'sftz109', 'sftz110', 'sftz111', 'sftz112', 'sftz113', 'sftz114']
        d_col_drops = ['id', 'relevance', 'search_term', 'product_title', 'product_description', 'search_and_prod_info',
                       'product_info', 'attr', 'brand', 'len_of_query', 'len_of_brand', 'query_in_description',
                       'word_in_brand', 'ratio_brand', 'ft04', 'ft05', 'ftx05', 'ftz04', 'ftz05', 'sft05', 'sftx05',
                       'sftz05', 'ft06', 'ft07', 'ft08', 'ft09', 'ft10', 'ft11', 'ft12', 'ft13', 'ft14', 'ftx06',
                       'ftx07', 'ftx08', 'ftx09', 'ftx10', 'ftx11', 'ftx12', 'ftx13', 'ftx14', 'ftz06', 'ftz07',
                       'ftz08', 'ftz09', 'ftz10', 'ftz11', 'ftz12', 'ftz13', 'ftz14', 'sft06', 'sft07', 'sft08',
                       'sft09', 'sft10', 'sft11', 'sft12', 'sft13', 'sft14', 'sft101', 'sft102', 'sft103', 'sft104',
                       'sft105', 'sft106', 'sft107', 'sft108', 'sft109', 'sft110', 'sft111', 'sft112', 'sft113',
                       'sft114', 'sftx06', 'sftx07', 'sftx08', 'sftx09', 'sftx10', 'sftx11', 'sftx12', 'sftx13',
                       'sftx14', 'sftx101', 'sftx102', 'sftx103', 'sftx104', 'sftx105', 'sftx106', 'sftx107', 'sftx108',
                       'sftx109', 'sftx110', 'sftx111', 'sftx112', 'sftx113', 'sftx114', 'sftz06', 'sftz07', 'sftz08',
                       'sftz09', 'sftz10', 'sftz11', 'sftz12', 'sftz13', 'sftz14', 'sftz101', 'sftz102', 'sftz103',
                       'sftz104', 'sftz105', 'sftz106', 'sftz107', 'sftz108', 'sftz109', 'sftz110', 'sftz111',
                       'sftz112', 'sftz113', 'sftz114']
        hd_searches = hd_searches.drop(d_col_drops, axis=1).values
        print 'tag1'
        return hd_searches


class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        print 'tag2'
        return data_dict[self.key].apply(unicode)


xgb_model = xgb.XGBRegressor(learning_rate=0.05,
                             silent=1,
                             objective="reg:linear",
                             nthread=-1,
                             gamma=0.5,
                             min_child_weight=5,
                             max_delta_step=1,
                             subsample=0.7,
                             colsample_bytree=0.7,
                             colsample_bylevel=1,
                             reg_alpha=0.5,
                             reg_lambda=1,
                             scale_pos_weight=1,
                             base_score=0.5,
                             seed=0,
                             missing=None)
tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
tsvd = TruncatedSVD(n_components=50, random_state=2016)
tnmf = NMF(n_components=50, random_state=2016)
clf = pipeline.Pipeline([
    ('union', FeatureUnion(
            transformer_list=[
                ('cst', cust_regression_vals()),
                ('txt2', pipeline.Pipeline(
                        [('s2', cust_txt_col(key='search_and_prod_info')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                ('txt3', pipeline.Pipeline(
                        [('s3', cust_txt_col(key='search_and_prod_info')), ('tfidf3', tfidf), ('tnmf', tnmf)]))
            ],
            transformer_weights={
                'cst': 1.0,
                'txt2': 1.0,
                'txt3': 1.0
            },
            n_jobs = 1
    )),
    ('xgb_model', xgb_model)])
param_grid = {'xgb_model__n_estimators': [2000], 'xgb_model__max_depth': [10]}
model = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=1, cv=2, verbose=20, scoring=RMSE)
model.fit(train, y_train.values)
print("----------------------------------")
print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)
print(model.best_score_ + 0.452493857808)
print("----------------------------------")
for i in range(len(model.grid_scores_)):
    print(model.grid_scores_[i][0], model.grid_scores_[i][1])
    print(model.grid_scores_[i][2])
    print("----------------------------------")

y_pred = model.predict(test)

min_y_pred = min(y_pred)
max_y_pred = max(y_pred)
min_y_train = min(y_train.values)
max_y_train = max(y_train.values)
print(min_y_pred, max_y_pred, min_y_train, max_y_train)
for i in range(len(y_pred)):
    if y_pred[i] < 1.0:
        y_pred[i] = 1.0
    if y_pred[i] > 3.0:
        y_pred[i] = 3.0
        # y_pred[i] = min_y_train + (((y_pred[i] - min_y_pred)/(max_y_pred - min_y_pred))*(max_y_train - min_y_train))
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('single_0314_rmse5_01.csv', index=False)
print("--- Training & Testing: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
