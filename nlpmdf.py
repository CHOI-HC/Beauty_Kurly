#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from kurlyprs.kdb import selBT,selSQL
#%%
rows=selBT("breview",'limit 1,2','review')
print(rows)
# %%
# JDK 설치 
# https://www.oracle.com/java/technologies/downloads/#jdk19-windows
# pip install konlpy

from konlpy.tag import Kkma
#%%

#bow Bag of words
#%%
def makeWdict(senten):
    kkma=Kkma()
    pos=kkma.pos(senten)
    pocate=['NNG','NNP','NNB','NNM','VV','VA','VCP','VCN']
    wdict={}
    for po in pos:
        if(po[1] in pocate):
            print(po[0])
            if po[0] in wdict.keys():
                wdict[po[0]]+=1
            else:
                wdict[po[0]]=1
            wdf=pd.DataFrame(data=list(wdict.items())).set_index(0)
    return wdf

# %%
def reSEN(sen):
    pattern='[^ㄱ-힣a-zA-Z0-9 .!?]'
    return re.sub(pattern,'',sen)

rows=selBT("breview",'limit 20','review')
def bowDF(rows=[]):
    mdf=pd.DataFrame()
    for r in rows:
        sen=reSEN(r[0])
        sdf=makeWdict(sen)
        print(sdf.index)
        mdf=pd.merge(mdf,sdf,how='outer', left_index=True, right_index=True,)
    return mdf.fillna(0)
    

# sdf=makeWdict(rows[0][0])
# jsen=' '.join( list(sdf.index))
# jsen
#plt.imshow(mdf.values)
# %%
def makeCorpus(senten):
    kkma=Kkma()
    pos=kkma.pos(senten)
    pocate=['NNG','NNP','NNB','NNM','VV','VA','VCP','VCN']
    wdict={}
    for po in pos:
        if(po[1] in pocate):
            print(po[0])
            if po[0] in wdict.keys():
                wdict[po[0]]+=1
            else:
                wdict[po[0]]=1
            wdf=pd.DataFrame(data=list(wdict.items())).set_index(0)
    return wdf

def bowDF(rows=[]):
    mdf=pd.DataFrame()
    for r in rows:
        sen=reSEN(r[0])
        sdf=makeWdict(sen)
        print(sdf.index)
        
    return mdf.fillna(0)

# %%
