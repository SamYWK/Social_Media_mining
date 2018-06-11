# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 20:16:03 2018

@author: SamKao
"""

import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('CNN_Accuracy.csv', header = None).values
plt.plot(df1[:], 'r', label = 'cnn_results')
#plt.legend(loc= 'upper right')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('dnn.png')