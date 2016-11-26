#-*-coding:utf-8-*-
'''
Coding Just for Fun
Created by burness on 16/4/19.
'''
from gensim.models import Doc2Vec
model = Doc2Vec.load_word2vec_format()
ws1=set('simpson strong tie 12 gaug angl,100001,3.0,angl bracket'.split(' ,'))
ws2 = set('angl bracket'.split(' ,'))
# ws2=set('ot onli do angl make joint stronger they also provid more consist straight '
#         'corner simpson strong tie offer a wide varieti of angl in variou size and '
#         'thick to handl light duti job or project where a structur connect is need '
#         'some can be bent (skewed) to match the project for outdoor project or those '
#         'where moistur is present use our zmax zinc coat connector which provid extra '
#         'resist against corros (look for a ""z"" at the end of the model number) versatil '
#         'connector for variou 90 connect and home repair projectsstrong than angl nail or screw fasten alonehelp ensur joint are consist straight and strongdimensions: 3in. xbi 3in. xbi 1 1/2in. made from 12 gaug steelgalvan for extra corros resistanceinstal with 10 d common nail or #9 xbi 1 1/2in. strong drive sd screw'
model.n_similarity(ws1,ws2)