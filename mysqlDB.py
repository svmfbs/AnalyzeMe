# Python 3.7.2
# $ cd ~/sample
# $ . ./bin/activate
# $ deactivate
#
# setting of MySQL database connection module
# $ brew install mysql-connector-c
# $ cd /usr/local/bin
# $ vi mysql_config
# # Create options
# libs="-L$pkglibdir"  # change
# libs="libs -lmysqlclient -lssl -lcrypto" # change
# # save mysql_config file
# pip3 install mysqlclient

import MySQLdb
import os
import sys
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets

print ("Version:= {0}".format(sys.version_info))

try :
    # connect to database and generate cursor
    connection = MySQLdb.connect(
        host='localhost',
        user='root',
        password='password',
        db='sitepoint',
        charset='utf8'
    )
    cursor = connection.cursor()

    # initialize cursor
    cursor.execute("DROP TABLE IF EXISTS employees")

    # create table
    cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS sitepoint.employees (
        id int(11) NOT NULL AUTO_INCREMENT,
        name varchar(50),
        location varchar(50),
        PRIMARY KEY (id)
    ) ENGINE=InnoDB  DEFAULT CHARSET=UTF8MB4 AUTO_INCREMENT=5
    """
    )

    # add data into table
    cursor.execute(
    """
    INSERT IGNORE INTO sitepoint.employees (id, name, location) VALUES
        (1, 'Jasmine', 'Australia'),
        (2, 'Jay', 'Indita'),
        (3, 'Jim', 'Germany'),
        (4, 'Lesley', 'Scotland');
    """
    )

    # display all data in table
    cursor.execute(
    """
    SELECT * FROM sitepoint.employees
    """
    )

    for row in cursor:
        print (row)

except MySQLdb.Error as e:
    print ('MySQLdb.Error: ', e)

# save status of database
connection.commit()

# close database connection
connection.close()
