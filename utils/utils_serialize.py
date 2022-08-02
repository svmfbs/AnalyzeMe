""" This is serialization program """
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import sys

sys.dont_write_bytecode = True # dont make .pyc files


class SerializeUtils():
    def __init__(self, pickle_file):
        self.pickle_file = pickle_file

    def serialize(self, lst2d):
        with open(self.pickle_file, 'wb') as f:
            pickle.dump(lst2d, f)

    def deserialize(self):
        with open(self.pickle_file, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    print(sys.prefix)
    print(sys.path)
