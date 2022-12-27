#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from kurlyprs.kdb import selBT,selSQL
from kurlyprs.kreview import getReview,getTopReview,monthBeauty,monthReview
#%%
from konlpy.tag import Kkma

# 데이터 준비
rows=selBT("breview",'limit 1,2','review')
# 정제
def reSEN(sen):
    pattern='[^ㄱ-힣a-zA-Z0-9 .!?]'
    return re.sub(pattern,'',sen)

def makeSentence(senten):
    # 형태소 분석기 준비
    kkma=Kkma()
    pos=kkma.pos(senten) 
    pocate=['NNG','NNP','NNB','NNM','VV','VA','VCP','VCN']
    sentence=''
    wordlist=[]
    for po in pos:
        if(po[1] in pocate):
            wordlist.append(po[0])
            sentence=' '.join(wordlist)
    return sentence,wordlist

def getCorpus(rows=[]):
    corpus=[]
    cowordlist=[]
    for r in rows:
        sen=reSEN(r[0])
        sen,wlist=makeSentence(sen)
        corpus.append(sen)
        cowordlist.append(wlist)
    return corpus,cowordlist

# %%
cl,wl=getCorpus(rows)
print(cl)
print(wl)

monthBeauty(1)

# %%
trows=getTopReview(2)

for tr in trows:
    print(tr)
    revs=getReview(tr[0])
len(revs)
# %%
cl,wl=getCorpus(revs)
# %%
wl[:10]
# %%

import gensim
from gensim import corpora,models

def getTopicNo(ldaModel=None,corp0=''):
    topicPd=dict(list(ldaModel[corp0]))
    tpmax=max(topicPd,key=topicPd.get)
    return tpmax,topicPd[tpmax]


#사전(단어 dict) 만들기
def trainLDA(rows,notopic=4,npass=10):
    cl,wl=getCorpus(rows)
    dictionary=corpora.Dictionary(wl)
    dict(dictionary)
    # 사전기반의 bow 만들기 
    corpus=[dictionary.doc2bow(w) for w in wl]
    N_TOPIC=4
    N_PASS=10
    ldaModel=gensim.models.LdaModel(corpus,
                                    num_topics=N_TOPIC,
                                    id2word=dictionary,
                                    passes=N_PASS)
    topics=ldaModel.print_topics(num_topics=4)
    corTopics=[]
    for i,cor in  enumerate(corpus):
        corTopic=getTopicNo(ldaModel,cor)
        corTopics.append([cl[i],corTopic[0],corTopic[1]])
    return corTopics
        


# %%
rlda=trainLDA(rows)
rlda
# %%
mres=monthBeauty(1)

monthReview(mres[0][0])
# %%
