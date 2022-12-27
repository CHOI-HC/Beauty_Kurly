#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from kurlyprs.kdb import selBT,selSQL
from kurlyprs.kreview import getReview,getTopReview
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
# %%
# 많이 나온 단어가 주된 단어다
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(stop_words=['록시땅','컬리'],ngram_range=(2,3),tokenizer=lambda txt: txt.split())
X = cv.fit_transform(cl)
names=cv.get_feature_names_out()
cbow=X.toarray()
print('names:',names)
print('단어 인덱스:',cv.vocabulary_)
print('cbow:',cbow)
# %%
# TF_IDF
cl=['i am the boy in the house','i am the girl in the house']
from sklearn.feature_extraction.text import TfidfVectorizer
tv=TfidfVectorizer(stop_words=['록시땅','컬리'],ngram_range=(1,1),tokenizer=lambda txt: txt.split())
X = tv.fit_transform(cl)
names=tv.get_feature_names_out()
tbow=X.toarray()
# %%
print('names:',names)
print('단어 인덱스:',tv.vocabulary_)
print('cbow:',tbow)
# %%

cl=['i am the boy in the house','i am the girl in the house']

tv=TfidfVectorizer(stop_words=['록시땅','컬리'],ngram_range=(1,1),tokenizer=lambda txt: txt.split())
X = tv.fit_transform(cl)
names=tv.get_feature_names_out()
tbow=X.toarray()
print('names:',names)

print('cbow:',tbow)

cv=CountVectorizer(stop_words=['록시땅','컬리'],ngram_range=(1,1),tokenizer=lambda txt: txt.split())
X = cv.fit_transform(cl)
names=cv.get_feature_names_out()
cbow=X.toarray()
print('names:',names)

print('cbow:',cbow)
# %%
import gensim
# %%
trows=getTopReview(2)
for tr in trows:
    print(tr)
    revs=getReview(tr[0])
revs
# %%
