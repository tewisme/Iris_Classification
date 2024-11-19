import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import missingno as msno
import pandas as pd

def overview(data):
    print(data.info())
    print(data.nunique())
    print(data.describe())
    #----------------------
    null_info = pd.DataFrame({'Null count': data.isnull().sum(), 'Null ratio': data.isnull().sum()/len(data)})
    null_info.drop('Species', inplace=True)
    null_info.sort_values(by='Null ratio', ascending=False, inplace=True)
    print(null_info)
    #----------------------
    fig, ax = plt.subplots(figsize=(9,4))
    msno.matrix(data, ax=ax)
    plt.show()
    #----------------------
    print('Setosa: ', format((data.Species=='Iris-setosa').sum()/len(data)*100, '.2f'))
    print('Versicolor: ', format((data.Species=='Iris-versicolor').sum()/len(data)*100, '.2f'))
    print('Virginica: ', format((data.Species=='Iris-virginica').sum()/len(data)*100, '.2f'))

def visualization(data):
    data1 = data.copy()
    
    print(data1['SepalLengthCm'].value_counts(dropna=False))

def proc(data):
    #overview(data)
    visualization(data)
