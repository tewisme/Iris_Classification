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


def proc(data):
    data.drop(columns="Id", inplace=True)
    x = data.iloc[:, 0:4].values
    y = data.iloc[:, 4].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    Naive_Bayes(x_train, x_test, y_train, y_test)
