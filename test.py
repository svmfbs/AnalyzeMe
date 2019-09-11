import os
import subprocess
import sys
import csv
import math
import random
import json
import logging
import pickle
import pprint
import socket
import platform
import pkgutil
import calendar
import time
import re
import unittest
import pathlib
from datetime import datetime
from collections import defaultdict
from collections import OrderedDict
# from lxml import etree
import numpy as np
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
# /////////////////////////////////////////////////////////////////////////////
# [MyModule] //////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
import csvIO

print ("Version:= {0}".format(sys.version_info))

class Man:
    def __init__(self, name):
        self.name = name
        print ("Initialized !")

    def hello(self):
        print ("Hello " + self.name + "!")

    def goodbye(self):
        print ("Good-bye " + self.name + "!")

m = Man("David")
m.hello()
m.goodbye()

# /////////////////////////////////////////////////////////////////////////////
# [Data Preparation] //////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
header_skipped = False
file_csv = r"C:\Users\Hide\Python\n225.csv"

csv = csvIO.CsvIO(header_skipped)
csv2D = csv.read(file_csv)

values = list(range(0, len(csv.header)))
dct = dict(zip(csv.header, values))

# for k,v in sorted(dct.items(), key=lambda x:x[1]):
#     print (str(k)+": "+str(v))

print (csv.header)

# /////////////////////////////////////////////////////////////////////////////
# [matplotlib] ////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
x = np.random.randn(1000)
y = np.random.randn(1000)
plt.scatter(x,y)
plt.title("scatter")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
#lt.show()

plt.hist(x)
plt.xlabel("x")
plt.ylabel("frequency")
#plt.show()

t = np.linspace(-np.pi, np.pi, 1000)
x1 = np.sin(2*t)
x2 = np.cos(2*t)
fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

axL.plot(t,x1,linewidth=2)
axL.set_title("sin")
axL.set_xlabel("t")
axL.set_ylabel("x")
axL.set_xlim(-np.pi, np.pi)
axL.grid(True)

axR.plot(t,x2,linewidth=2)
axR.set_title("cos")
axR.set_xlabel("t")
axR.set_ylabel("x")
axR.set_xlim(-np.pi, np.pi)
axR.grid(True)
# fig.show()

plt.subplot(1,2,1)
plt.plot([0,1],[0,1],color="red")
plt.subplot(1,2,2)
plt.plot([0,1],[0,1],color="blue")
plt.show()
