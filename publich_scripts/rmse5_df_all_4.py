import time
start_time = time.time()
import pandas as pd
import re

def str_join_words(str1, str2):
    s=(" ").join(["q_"+ z for z in str1.split(" ")])  + " " + str2
    return s

df_all = pd.read_csv('df_all_rmse5_3.csv', index_col=0, encoding="ISO-8859-1")
print("--- Files Loaded: %s minutes ---" % round(((time.time() - start_time)/60),2))
df_all['search_and_prod_info'] = df_all['product_info'].map(lambda x:str_join_words(x.split('\t')[0],x.split('\t')[1]))
print("--- Feature Created: %s minutes ---" % round(((time.time() - start_time)/60),2))
df_all.to_csv('df_all_rmse5_4.csv')