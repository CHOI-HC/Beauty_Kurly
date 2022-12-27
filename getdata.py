#%%
# pip install -r requirements.txt
# pip install pandas bs4 selenium webdriver_manager
# %%

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import sqlite3
from kurlyprs.kprs import getHot,getAllRev
from kurlyprs.kdb import selBT


coption= webdriver.ChromeOptions()
driver = webdriver.Chrome(ChromeDriverManager().install(),options=coption)

#%%
# 리뷰 하단의 다음 버튼이 있으면 클릭
pdtcode=5161423
for i in range(1,3):
    driver=getHot(driver,i)
#getAllRev(driver,pdtcode)
#%%
rows=selBT('beauty','','pcode')
print(rows)
for row in rows:
    getAllRev(driver,row[0])


    
#
# %%
