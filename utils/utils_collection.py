""" This is collections program """
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from collections import Counter
from heapq import merge
sys.dont_write_bytecode = True # dont make .pyc files


class Collection:
    @classmethod
    def all_if(cls, cond, lst):
        return all(cond(x) for x in lst)

    @classmethod
    def any_if(cls, cond, lst):
        return any(cond(x) for x in lst)

    @classmethod
    def find_if(cls, cond, lst, default=None):
        return next((x for x in lst if cond(x)), default)

    @classmethod
    def merge_dict(cls, dict_src, **kwargs):
        return {**dict_src, **kwargs}

    @classmethod
    def merge_list(cls, lst_src, lst_add, is_uniq=False):
        merged = list(merge(lst_src, lst_add))
        if is_uniq:
            return cls.distinct(merged)
        else:
            return merged

    @classmethod
    def distinct(cls, lst):
        seen = set()
        seen_add = seen.add
        return [x for x in lst if x not in seen and not seen_add(x)]

    @classmethod
    def make_word_list(cls, str_eng, del1=',', del2='.'):
        s_remove = str_eng.replace(del1, '').replace(del2, '')
        return s_remove.split()

    @classmethod
    def flatten_with_2_depth(cls, nested_list):
        """ flatten list with 2-depth """
        return [e for inner_list in nested_list for e in inner_list]

    @classmethod
    def flatten_with_any_depth(cls, nested_list):
        """ flatten nested list in manner of depth-first search """
        # prepare flat list and fringe
        flat_list = []
        fringe = [nested_list]
        while len(fringe) > 0:
            node = fringe.pop(0)
            # add child element into fringe when node is list, otherwise add as it is
            if isinstance(node, list):
                fringe = node + fringe
            else:
                flat_list.append(node)
        return flat_list


if __name__ == '__main__':
    print(os.getcwd())
    print(f'# File name: {__file__}')

    some_list = ['foo', 'bar', 'baz']
    print(Collection.find_if(lambda l: l == 3, [1, 2, 3, 4]))
    print(Collection.find_if(lambda l: l == 3, [1, 2, 4], default='noooo'))
    print('# --------------------')
    print(Collection.all_if(lambda x: len(x) == 3, some_list))
    print(Collection.any_if(lambda x: x == 10, range(10)))

    d1 = {'a': 1, 'e': 5}
    d2 = {'b': 2}
    d3 = {'c': 3, 'd': 4, 'f': 6}
    d4 = {}
    d_merged1 = {**d1, **d2, **d3, **d4}
    d_merged2 = Collection.merge_dict(d1, **d2, **d3, **d4)
    print(d_merged1)
    print(d_merged2)

    some_list2 = [1, 2, 3, 4, 7, 8, 5, 1]
    some_list3 = [4, 5, 6]
    print(some_list.count('foo'))
    print(some_list.count('boo'))
    print(Collection.merge_list(some_list2, some_list3))
    print(Collection.merge_list(some_list2, some_list3, True))
    li = [3, 4, 3, 2, 5, 4]
    print(Collection.distinct(li))

    lst = ['a', 'a', 'a', 'a', 'b', 'c', 'c']
    cnt = Counter(lst)
    print(type(cnt))
    print(cnt)
    print(cnt.keys())
    print(cnt.values())
    print(cnt.items())
    print(cnt.most_common())
    print(cnt.most_common()[::-1])

    s = 'government of the people, by the people, for the people.'
    word_list = Collection.make_word_list(s)
    print(word_list)
    print(word_list.count('people'))
    c = Counter(word_list)
    print(c.most_common())

    a = [[1, 3], [5, 7], [9]]
    b = [[1, 3], [[5]], [[7], 9]]
    print(Collection.flatten_with_2_depth(a))
    print(Collection.flatten_with_any_depth(b))
