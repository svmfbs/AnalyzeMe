""" This is xml and json program """
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import xml.etree.ElementTree as ET
sys.dont_write_bytecode = True # dont make .pyc files

PATH_DATA = r'/Users/sample/data'
PATH_WORD = r'/Users/sample/word'
PATH_NIFI = r'/Users//NiFi/'
PATH_PROJ = r'/Users/sample/proj'

class JsonUtils:
    def __init__(self, jsonin, jsonout=None):
        self.file_in = jsonin
        self.file_out = jsonout
        self.json_in = open(jsonin, 'r') if jsonin is not None else None
        self.json_out = open(jsonout, 'w') if jsonout is not None else None

    def to_dict(self):
        return json.load(self.json_in)

    def to_string(self, json_dict):
        return json.dumps(json_dict)

    def to_dict_from_str(self, json_str):
        return json.loads(json_str)

    def to_write(self, json_dict, indentnb=4):
        json.dump(json_dict, self.json_out, indent=indentnb)

class XmlUtils:
    def __init__(self, xmlin):
        self.file_in = xmlin
        self.tree = ET.parse(self.file_in)
        self.root = self.tree.getroot()

    def get_all_item1(self, item1):
        items = list()
        for child in self.root:
            if child.find(item1) is not None:
                items.append(child.find(item1).text)
        return items

    def get_all_item2(self, item1, item2):
        items = list()
        for child in self.root:
            if child.find(item1) is not None:
                if child.find(item2) is not None:
                    text1 = child.find(item1).text
                    text2 = child.find(item2).text
                    items.append([text1, text2])
        return items


def main_json():
    json_only = 'test.json'
    json2_only = 'test2.json'
    json_path = os.path.join(PATH_DATA, json_only)
    json_path2 = os.path.join(PATH_DATA, json2_only)

    # read json
    js = JsonUtils(json_path, json_path2)
    json_dict = js.to_dict()
    print(f'json_dict:{type(json_dict)}')
    print(f'json_dict:{json_dict}')
    json_str = js.to_string(json_dict)
    print(f'json_str:{type(json_str)}')
    print(f'json_str:{json_str}')
    # read string
    json_dict2 = js.to_dict_from_str(json_str)
    print(f'json_dict2:{type(json_dict2)}')
    print(f'json_dict2:{json_dict2}')
    # write json
    js.to_write(json_dict2)


def main_xml():
    xml_only = 'words.xml'
    xml_path = os.path.join(PATH_WORD, xml_only)

    xu = XmlUtils(xml_path)
    print(f'{xu.root.tag}, {xu.root.attrib}')

    # items1 = xu.get_all_item1('WORD')
    items2 = xu.get_all_item2('WORD', 'MEANING')
    for x in items2:
        print(f'{x[0]},{x[1]}')


if __name__ == "__main__":
    # main_json()
    # main_xml()
    # json_path = os.path.join(PATH_NIFI, 'test')
    json_path = os.path.join(PATH_PROJ, 'daily.json')
    # [read json]
    js = JsonUtils(json_path)
    print(js.to_dict())
