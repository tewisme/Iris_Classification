import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import missingno as msno

#Data Processing

data_path = '..\\data\\iris.csv'
data = pd.read_csv(data_path)
data.head()

#data.info(verbose=True) how many column, line/column
#print(data.nunique()) differant value on each column
#print(data.describe())

"""
# Count null-value, null-ratio
null_info = pd.DataFrame({'Null count': data.isnull().sum(), 'Null ratio': data.isnull().sum()/len(data)})
null_info.drop('Species', inplace=True)
null_info.sort_values(by='Null ratio', ascending=False, inplace=True)
print(null_info)
"""

"""
# represent null throught image
msno.matrix(data)
plt.show()
"""
