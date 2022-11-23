# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 01:39:15 2022

@author: jf3g19
"""

# https://www.oreilly.com/library/view/python-cookbook/0596001673/ch05s17.html
class Options:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __lshift__(self, other):
        """ overloading operator << """
        s = self.__copy__()
        s.__dict__.update(other.__dict__)
        return s

    def __copy__(self):
        return self.__class__(**self.__dict__)
    
    def as_string(self):
        str_array = []
        for key,value in self.__dict__.items():
            str_array.append(f'\{key}={value}')
        return ','.join(str_array)