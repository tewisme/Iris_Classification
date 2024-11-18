import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import missingno as msno


def proc(data):
    #print(data)

    #Overview 'bout Data

    #data.info(verbose=True) #how many column, line/column
    #print(data.nunique()) #differant value on each column
    #print(data.describe()) #absolutly for describe data

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
    
    """
    # Flowers ratio
    print('% Setosa:', format((data.Species=='Iris-setosa').sum()/len(data)*100, '.2f'))
    print('% Versicolor:', format((data.Species=='Iris-versicolor').sum()/len(data)*100, '.2f'))
    print('% Virginica:', format((data.Species=='Iris-virginica').sum()/len(data)*100, '.2f'))
    """
    
    #Represent data and change data throught data.copy()
    
    data1 = data.copy()
    
    #SepalLength Begin
    
    #print(data1['SepalLengthCm'].value_counts(dropna=False)) #dropna = Drop Nan: Loai bo non-value, dropna=False => khong loai bo non-value; defaul: dropna=True
    
    plt.figure(figsize=(9,4))
    #1st subplot
    plt.subplot(1,2,1)
    sns.histplot(data1.SepalLengthCm, bins=20, kde=True)
    plt.title('Histplot diagram for SepalLengthCm')
    
    #2nd subplot
    plt.subplot(1,2,2)
    sns.boxplot(x="Species", y="SepalLengthCm", data=data1)
    sns.stripplot(x='Species', y='SepalLengthCm', data=data1, jitter=True, edgecolor='grey')
    plt.title('Boxplot for SepalLengthCm')
    
    plt.show()
    #SepalLengh End
    
    #SepalWidth Begin
    
    plt.figure(figsize=(9,4))
    
    plt.show()
    #SepalWidth End
