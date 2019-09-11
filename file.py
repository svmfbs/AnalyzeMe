import sys
import os
import urllib3
from bs4 import BeautifulSoup
import certifi
import numpy as np
import pandas as pd

print ("Hello World !!")

# Display macOS environment variables
paths = os.environ['PATH'].split(':')
# print ('\n'.join(paths))

# Find out your current working directory
current_dir = os.getcwd()
print ("Current working directory: ", current_dir)

# Display all of the files found in your current working directory
all_files_curr_dir = os.listdir(current_dir)
print ("Files:= ",all_files_curr_dir)

arr = np.asarray([1,2,3,4,5], dtype=np.float32)
# print (arr)
# print (arr.shape)
# print ()

df = pd.read_csv("KEN_ALL_ROME_UTF8.CSV")
# print (df.head(3))
# print ()
# print (df.tail())
print ('dataframeの行数・列数の確認==>',df.shape)

# http = urllib3.PoolManager()
# r=http.request('GET','http://httpbin.org/get')
# print ("HTTP Status:= ", r.status)
# print (r.data)

# アクセスするURL
domain_au = 'https://www.au.com'
domain_auone = 'https://portal.auone.jp'
domain_nupl = 'http://www.nupl.info/cont_1'
domain_nikkei = "https://www.nikkei.com/"
url = domain_nikkei

# httpsの証明書検証を実行してみる
http = urllib3.PoolManager(
    cert_reqs='CERT_REQUIRED',
    ca_certs=certifi.where()
)
r = http.request('GET', url)

soup = BeautifulSoup(r.data, 'html.parser')

# タイトル要素を取得する
title_tag = soup.title

# 要素の文字列を取得する
title = title_tag.string

print (title_tag)
print (title)