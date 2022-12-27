import sqlite3
dbfile='data\kurly.db'

#data=('hot','1000074086', '10', '[아틀리에 코롱] 클레망틴 캘리포니아 코롱 압솔뤼 30ml', '15', '97750', '115000')
def insBT(data):
    
    inssql='insert into beauty (cate,pcode,coupon,title,sale,price,oriprice) \
        values(?,?,?,?,?,?,?)'
    with sqlite3.connect(dbfile) as conn:
        cur=conn.cursor()
        cur.execute(inssql,data)
        
## review insert        
def insRev(data):
    inssql='insert into breview (pcode,rkey,review,date) values(?,?,?,?)'
    with sqlite3.connect(dbfile) as conn:
        cur=conn.cursor()
        cur.execute(inssql,data)

#셀렉트하기
def selSQL(sql=''):
    with sqlite3.connect(dbfile) as conn:
        cur=conn.cursor()
        cur.execute(sql)
        rows=cur.fetchall()
    return rows
  
# 컬리 셀렉트하기
def selBT(tbl='beauty',whr='where pcode=',cols='*'):
    sql='select '+cols+' from '+tbl+' '+whr
    return selSQL(sql)


