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
    
    #Sepal Length
    #print(data1['SepalLengthCm'].value_counts(dropna=False))
    #"""
    plt.figure(figsize=(9,4))
    #first subplot
    plt.subplot(1,2,1)
    sns.histplot(data1.SepalLengthCm, bins=10, kde=True)
    plt.title('Histplot diagram for SepalLengthCm')
    #second subplot
    plt.subplot(1,2,2)
    sns.boxplot(x="Species", y="SepalLengthCm", data=data1)
    sns.stripplot(x="Species", y="SepalLengthCm", data=data1, jitter=True, edgecolor='black')
    plt.title("Boxplot for SepalLengthCm")
    #"""

    #Sepal Width
    #print(data1['SepalWidthCm'].value_counts(dropna=False))
    #"""
    plt.figure(figsize=(9,4))
    #first subplot
    plt.subplot(1,2,1)
    sns.histplot(data1.SepalWidthCm, bins=5, kde=True)
    plt.title("Histplot diagram for SepalWidthCm")
    #second subplot
    plt.subplot(1,2,2)
    sns.violinplot(x="Species", y="SepalWidthCm", data=data1)
    sns.stripplot(x="Species", y="SepalWidthCm", data=data1, jitter=True, edgecolor='black')
    plt.title("Boxplot for SepalWidthCm")
    #"""
    
    #Petal Length
    #print(data1['PetalLengthCm'].value_counts(dropna=False))
    #"""
    #first subplot
    plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    sns.histplot(data1.PetalLengthCm, bins=10, kde=True)
    plt.title("Histplot diagram for PetalLengthCm")
    #second subplot
    plt.subplot(1,2,2)
    sns.boxplot(x="Species", y="PetalLengthCm", data=data1)
    sns.stripplot(x="Species", y="PetalLengthCm", data=data1, jitter=True, edgecolor='black')
    plt.title("Boxplot for PetalLengthCm")
    #"""

    #Petal Width
    #print(data1['PetalWidthCm'].value_counts(dropna=False))
    #"""
    plt.figure(figsize=(9,4))
    #first subplot
    plt.subplot(1,2,1)
    sns.histplot(data1.PetalWidthCm, bins=10, kde=True)
    plt.title('Histplot diagram for PetalWidthCm')
    #second subplot
    plt.subplot(1,2,2)
    sns.violinplot(x="Species", y="PetalWidthCm", data=data1)
    sns.stripplot(x="Species", y="PetalWidthCm", data=data1, jitter=True, edgecolor='black')
    plt.title('Boxplot for PetalWidthCm')
    #"""

    plt.show()


def relationship(data):
    data1 = data.copy()
    
    #sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=data1, size=5, hue="Species")
    #plt.title("Relationship between SepalLengthCm and SepalWidthCm")
    #------------------
    #sns.set_style('whitegrid')
    #sns.lmplot(x='SepalLengthCm', y='SepalWidthCm', data=data1, hue='Species', markers=['o', 'v', 'x'])
    #plt.title("Regression between SepalLengthCm and SepalWidthCm")
    #------------------
    #sns.jointplot(x="PetalLengthCm", y="PetalWidthCm", data=data1, size=5, hue="Species")
    #plt.title("Relationship between PetalLengthCm and PetalWidthCm")
    #------------------
    #sns.set_style('whitegrid')
    #sns.lmplot(x='PetalLengthCm', y='PetalWidthCm', data=data1, hue="Species", markers=['o', 'v', 'x'])
    #plt.title("Regression between PetalLengthCm and PetalWidthCm")

    #plt.figure(figsize=(12,6)) #palette must beside hue
    #plt.subplot(2,2,1); sns.barplot(x='Species', y='SepalLengthCm', data=data1, palette='cubehelix', hue='Species')
    #plt.subplot(2,2,2); sns.barplot(x='Species', y='SepalWidthCm', data=data1, palette='Oranges', hue='Species')
    #plt.subplot(2,2,3); sns.barplot(x='Species', y='PetalLengthCm', data=data1, palette='Oranges', hue='Species')
    #plt.subplot(2,2,4); sns.barplot(x='Species', y='PetalLengthCm', data=data1, palette='cubehelix', hue='Species')
    
    #plt.figure(figsize=(12,6))
    #sns.heatmap(data1.drop('Species', axis=1).corr(), annot=True, cmap='Dark2_r', linewidths=2)

    plt.show()


def summarize(data):
    data1 = data.copy()

    #sns.pairplot(data=data1, kind='scatter')
    sns.pairplot(data=data1, hue='Species')

    plt.show()


def proc(data):
    #overview(data)
    #visualization(data)
    relationship(data)
    #summarize(data)
