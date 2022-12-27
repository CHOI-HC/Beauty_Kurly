# 컬리의 핫아이템 가져오기

import time
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from kurlyprs.kdb import insBT,insRev,selBT

def getItem(elem):
    pdtcode=elem.attrs['href'].replace('/goods/','')
    
    couponElem=elem.select('.css-w9ezyo')
    #print(couponElem)
    if(len(couponElem)>0):
        coupon=couponElem[0].text.replace('%쿠폰','').replace('+','').strip()
    else:
        coupon=0
    # 타이틀
    titleElem=elem.select('.css-rklo75')
    if(len(titleElem)>0):
        title=titleElem[0].text
    else:
        title='no name'
    # 디스카운트        
    discElem=elem.select('.discount-rate')
    if(len(discElem)>0):        
        disc=discElem[0].text.replace('%','').strip()
    else:
        disc=0
    # 판매가
    priceElem=elem.select('.sales-price')
    if(len(priceElem)>0):
        price=priceElem[0].text.replace('원','').replace(',','').strip()
    else:
        price=0
    # 정가
    oriElem=elem.select('.dimmed-price')
    if(len(oriElem)>0):
        oriprice=oriElem[0].text.replace('원','').replace(',','').strip()
    else:
        oriprice=price
    return ('hot',pdtcode,coupon,title,disc,price,oriprice)
#


#%%
# 목표 요소에 키값 입력하기
def kSendkey(driver=None ,_selector='',kv=Keys.ENTER):
    src=driver.find_elements(by=By.CSS_SELECTOR,value=_selector)
    src[0].send_keys(kv)
    
def kTarget(driver=None ,_selector=''):#.click() 을 위한 타겟
    target=driver.find_element(by=By.CSS_SELECTOR,value=_selector)
    return target
        
def kgetBS(thtml='',_tgtsel=''):# bs4를 위한 함수
    soup=bs(thtml,'html.parser')
    elems=soup.select(_tgtsel)
    return elems

# 열린 url로 부터 원하는 요소 반환하기    
def kgetElem(driver,_selector='.css-vjtyom',_tgtsel='div > a'):
    src=driver.find_elements(by=By.CSS_SELECTOR,value=_selector)
    tgtHtml=src[0].get_attribute('outerHTML')
    elems =kgetBS(tgtHtml,_tgtsel)
    return elems

# 스크래이핑을 위해서 url 열기
def kScrap(driver=None,_srcurl= '',_selector='.css-vjtyom',_tgtsel='div > a'):
    driver.implicitly_wait(10)
    driver.get(_srcurl)
    elems=kgetElem(driver,_selector,_tgtsel)
    return (elems,driver)

# 크롤링하여 DB에 넣기
def kinsDB(elems):
    for elem in elems:
        data=getItem(elem)
        print('EEEEEEEEEE',data)
        rows=selBT('beauty','where pcode='+data[1])
        if(rows):
            print('이미 존재함 beauty')
        else:
            insBT(data)
        print(".",end='')
        
# 컬리로부터 데이터를 수집하여 DB 입력
def getHot(driver=None,pno=1):
    url='https://www.kurly.com/collections/beauty-nowhot?site=beauty&page={}'.format(pno)
    sel='.css-vjtyom'
    tsel='div > a'
    elems,driver=kScrap(driver,url,sel,tsel)
    #####################################
    kinsDB(elems)
    return driver
#%%
# 버튼찾기
def getPdt(driver=None,pcode='5161423'):
    # pcode 받아서 창을열고 리뷰클릭
    url='https://www.kurly.com/goods/'+str(pcode)
    sel='.css-tse2s2'
    tsel='li > a'
    elems,driver=kScrap(driver,url,sel,tsel)
    #time.sleep(2)
    print(len(elems))
    # 후기클릭
    tgtSel='#top > div.css-n48rgu.ex9g73v0 > div.css-16c0d8l.e1brqtzw0 > nav > ul > li:nth-child(3) > a'
    kTarget(driver,tgtSel).click()
    time.sleep(2)
    return driver
    

    
def getReview(pdtcode=5161423,driver=None):
    sel='.css-1nrf0nk'
    tsel='div.css-169773r'
    elems=kgetElem(driver,sel,tsel)
    print('****',len(elems))
    for elem in elems:
        #print(elem)
        rev=elem.select('article p')[0].text
        user=elem.select('span.css-f3vz0n')[0].text[0]
        day=elem.select('footer span.css-14kcwq8')[0].text
        revkey=user+day+'_'+str(len(rev))
        data=(pdtcode,revkey,rev,day)
        rows=selBT('breview','where rkey="'+data[1]+'"','rkey')
        if(rows):
            print('x',end='')
        else:
            insRev(data)
        print('.',end='')
        #print(kgetBS(elem,'p'))
        
def getAllRev(driver=None,pdtcode=5161423):
    print('*****',pdtcode,'*********')
    driver=getPdt(driver,pdtcode)
    nextButtonCss='#review > section > div.css-1nrf0nk.e1kog1is13 > div.css-jz9m4p.e1kog1is5 > button.css-1orps7k.e1kog1is1'
    nextButton=kTarget(driver,nextButtonCss)
    #print('next button:',nextButton.get_attribute("disabled"))

    while (nextButton.get_attribute("disabled")==None):
        #print(nextButton.get_attribute("disabled"))
        getReview(pdtcode,driver)
        #print('-'*30)
        nextButton.click()
        time.sleep(2)
    getReview(pdtcode,driver)
    

        