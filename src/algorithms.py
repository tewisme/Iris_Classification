#lib
#metrics
from sklearn.metrics import make_scorer, accuracy_score, precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, f1_score
#model selection
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier     
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
#model support
from sklearn.preprocessing import LabelEncoder

def Naive_Bayes(x_train, x_test, y_train, y_test):
    gaussian = GaussianNB()
    gaussian.fit(x_train, y_train)
    y_pred = gaussian.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_nb = round(accuracy*100, 2)
    acc_gaussian = round(gaussian.score(x_train, y_train)*100, 2)
    print(accuracy, accuracy_nb)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')

def Dec_Tree(x_train, x_test, y_train, y_test):
    dec_tree = DecisionTreeClassifier()
    dec_tree.fit(x_train, y_train)
    y_pred = dec_tree.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_dt = round(accuracy*100, 2)
    print(accuracy, accuracy_dt)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')

def Random_Forest(x_train, x_test, y_train, y_test):
    rd_forest = RandomForestClassifier(n_estimators=100)
    rd_forest.fit(x_train, y_train)
    y_pred = rd_forest.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_rf = round(accuracy*100, 2)
    print(accuracy, accuracy_rf)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    racall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')

def KNN(x_train, x_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_knn = round(accuracy*100, 2)
    print(accuracy, accuracy_knn)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')

def MLP(x_train, x_test, y_train, y_test):
    mlp = MLPClassifier(random_state=42, max_iter=1000)
    mlp.fit(x_train, y_train)
    y_pred = mlp.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_mlp = round(accuracy*100, 2)
    print(accuracy, accuracy_mlp)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')


def proc(data):
    data.drop(columns="Id", inplace=True)
    x = data.iloc[:, 0:4].values
    y = data.iloc[:, 4].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    #Naive_Bayes(x_train, x_test, y_train, y_test)
    #Dec_Tree(x_train, x_test, y_train, y_test)
    #Random_Forest(x_train, x_test, y_train, y_test)
    #KNN(x_train, x_test, y_train, y_test)
    #MLP(x_train, x_test, y_train, y_test)
