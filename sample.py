# Python 3.7.2
# $ cd ~/sample
# $ . ./bin/activate
# $ deactivate

import MySQLdb
import os
import sys
import math
import urllib.request
import requests
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
# [MyModele] //////////////////////////
import csvIO

a = 3
b = 8
c = a + b
print (c)

#pi = math.pi
#x = np.linspace(0, 2*pi, 100)
# y = np.sin(x)
# plt.plot(x, y)
# plt.show()

header_skipped = False
file_csv = r"/Users/hide/sample/KEN_ALL_ROME_UTF8.csv"
csv = csvIO.CsvIO(header_skipped)
csv2D = csv.read(file_csv)
nums = list(range(0, len(csv.header)))
dct = dict(zip(csv.header, nums))
for k, v in sorted(dct.items(), key=lambda x:x[1]):
    print (str(v) + " : " + str(k))
print (csv.header)
print ()

header_name = csv.header[1]
prefectures = csv.get_column_data(csv.header[1])

# make list with unique element without ordering
prefectures_unique = list(set(prefectures))
print (prefectures_unique)
print (len(prefectures_unique))
#print (csv2D[0][0])

# make list with duplicated element with ordering
l_duplicate_order = [x for x in dict.fromkeys(prefectures) if prefectures.count(x)>1]
print (l_duplicate_order)
print ("Count:= ", len(l_duplicate_order))

# read URL and get headers which include cookie so on
domain_au = 'https://www.au.com'
domain_auone = 'https://portal.auone.jp'
domain_nupl = 'http://www.nupl.info/cont_1'

get_url_info = requests.get(domain_au)
status_code = get_url_info.status_code
print (status_code)
dict_url = dict(get_url_info.headers)
for k, v in sorted(dict_url.items(), key=lambda x:x[1]):
    print (k + " : " + v)
print()

f = urllib.request.urlopen(domain_nupl)
print (f.read())
