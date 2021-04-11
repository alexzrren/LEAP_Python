# -*- coding: utf-8 -*-
"""
@Author  : Zirui Ren
@Time    : 2021/4/1 20:15
@Contact : 1131061444@qq.com/renzirui@webmail.hzau.edu.cn
@File    : MatxToTriplet.py
@Desc: PyCharm
"""
import pandas as pd
import numpy as np


'''
Matx:关系矩阵，必须满足 行数==列数
genelist:包含基因名的list或者pandas Series, 长度必须和关系矩阵的行、列数相等，并且排列和关系矩阵的index和column排列相同
'''
def MatxToTriplet(Matx, genelist):
    shape = Matx.shape[0]
    Triplet = []
    for row in range(shape):
        for col in range(shape):
            Triplet.append([genelist[row], genelist[col], Matx[row, col]])
    TripletMatx = pd.DataFrame(Triplet, columns=['gene1', 'gene2', 'value'])
    return TripletMatx