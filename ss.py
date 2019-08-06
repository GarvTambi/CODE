Python 2.7.16 (v2.7.16:413a49145e, Mar  4 2019, 01:37:19) [MSC v.1500 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> print("hello world")
hello world
>>> import numpy
>>> import scipy
>>> import matplotlib
>>> import scipy
>>>  print('scipy: {}'.format(scipy.__version__))
 
  File "<pyshell#5>", line 2
    print('scipy: {}'.format(scipy.__version__))
    ^
IndentationError: unexpected indent
>>> print('scipy: {}'.format(scipy.__version__))
scipy: 1.2.2
>>> import pandas
>>> print('pandas: {}'.format(pandas.__version__))
pandas: 0.24.2
>>> import pandas
>>> from pandas.plotting import scatter_matrix
>>> import matplotlib.pyplot as plt
>>> from sklearn import model_selection
from sklear
>>> from sklearn.metrics import classification_report
>>> from sklearn.metrics import confusion_matrix
>>> from sklearn.metrics import accuracy_store

Traceback (most recent call last):
  File "<pyshell#15>", line 1, in <module>
    from sklearn.metrics import accuracy_store
ImportError: cannot import name accuracy_store
>>> from sklearn.metrics import accuracy_score
>>> from sklearn.linear_model import logisticRegression

Traceback (most recent call last):
  File "<pyshell#17>", line 1, in <module>
    from sklearn.linear_model import logisticRegression
ImportError: cannot import name logisticRegression
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.tree import DecisionTreeClassifier
>>> from sklearn.neighbors import KNeighborsClassifier
>>> from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
>>> from sklearn.naive_bayes import GaussianNB
>>> from sklearn.svm import SVC
>>> url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
>>> print(dataset.shape)

Traceback (most recent call last):
  File "<pyshell#25>", line 1, in <module>
    print(dataset.shape)
NameError: name 'dataset' is not defined
>>> print(dataset.shape)

Traceback (most recent call last):
  File "<pyshell#26>", line 1, in <module>
    print(dataset.shape)
NameError: name 'dataset' is not defined
>>> url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
>>> names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
>>> dataset = pandas.read_csv(url, names=names)
>>> print(dataset.shape)
(150, 5)
>>> print(dataset.head(30))
    sepal-length  sepal-width  petal-length  petal-width        class
0            5.1          3.5           1.4          0.2  Iris-setosa
1            4.9          3.0           1.4          0.2  Iris-setosa
2            4.7          3.2           1.3          0.2  Iris-setosa
3            4.6          3.1           1.5          0.2  Iris-setosa
4            5.0          3.6           1.4          0.2  Iris-setosa
5            5.4          3.9           1.7          0.4  Iris-setosa
6            4.6          3.4           1.4          0.3  Iris-setosa
7            5.0          3.4           1.5          0.2  Iris-setosa
8            4.4          2.9           1.4          0.2  Iris-setosa
9            4.9          3.1           1.5          0.1  Iris-setosa
10           5.4          3.7           1.5          0.2  Iris-setosa
11           4.8          3.4           1.6          0.2  Iris-setosa
12           4.8          3.0           1.4          0.1  Iris-setosa
13           4.3          3.0           1.1          0.1  Iris-setosa
14           5.8          4.0           1.2          0.2  Iris-setosa
15           5.7          4.4           1.5          0.4  Iris-setosa
16           5.4          3.9           1.3          0.4  Iris-setosa
17           5.1          3.5           1.4          0.3  Iris-setosa
18           5.7          3.8           1.7          0.3  Iris-setosa
19           5.1          3.8           1.5          0.3  Iris-setosa
20           5.4          3.4           1.7          0.2  Iris-setosa
21           5.1          3.7           1.5          0.4  Iris-setosa
22           4.6          3.6           1.0          0.2  Iris-setosa
23           5.1          3.3           1.7          0.5  Iris-setosa
24           4.8          3.4           1.9          0.2  Iris-setosa
25           5.0          3.0           1.6          0.2  Iris-setosa
26           5.0          3.4           1.6          0.4  Iris-setosa
27           5.2          3.5           1.5          0.2  Iris-setosa
28           5.2          3.4           1.4          0.2  Iris-setosa
29           4.7          3.2           1.6          0.2  Iris-setosa
>>> print(dataset.head(150))
     sepal-length  sepal-width  petal-length  petal-width           class
0             5.1          3.5           1.4          0.2     Iris-setosa
1             4.9          3.0           1.4          0.2     Iris-setosa
2             4.7          3.2           1.3          0.2     Iris-setosa
3             4.6          3.1           1.5          0.2     Iris-setosa
4             5.0          3.6           1.4          0.2     Iris-setosa
5             5.4          3.9           1.7          0.4     Iris-setosa
6             4.6          3.4           1.4          0.3     Iris-setosa
7             5.0          3.4           1.5          0.2     Iris-setosa
8             4.4          2.9           1.4          0.2     Iris-setosa
9             4.9          3.1           1.5          0.1     Iris-setosa
10            5.4          3.7           1.5          0.2     Iris-setosa
11            4.8          3.4           1.6          0.2     Iris-setosa
12            4.8          3.0           1.4          0.1     Iris-setosa
13            4.3          3.0           1.1          0.1     Iris-setosa
14            5.8          4.0           1.2          0.2     Iris-setosa
15            5.7          4.4           1.5          0.4     Iris-setosa
16            5.4          3.9           1.3          0.4     Iris-setosa
17            5.1          3.5           1.4          0.3     Iris-setosa
18            5.7          3.8           1.7          0.3     Iris-setosa
19            5.1          3.8           1.5          0.3     Iris-setosa
20            5.4          3.4           1.7          0.2     Iris-setosa
21            5.1          3.7           1.5          0.4     Iris-setosa
22            4.6          3.6           1.0          0.2     Iris-setosa
23            5.1          3.3           1.7          0.5     Iris-setosa
24            4.8          3.4           1.9          0.2     Iris-setosa
25            5.0          3.0           1.6          0.2     Iris-setosa
26            5.0          3.4           1.6          0.4     Iris-setosa
27            5.2          3.5           1.5          0.2     Iris-setosa
28            5.2          3.4           1.4          0.2     Iris-setosa
29            4.7          3.2           1.6          0.2     Iris-setosa
..            ...          ...           ...          ...             ...
120           6.9          3.2           5.7          2.3  Iris-virginica
121           5.6          2.8           4.9          2.0  Iris-virginica
122           7.7          2.8           6.7          2.0  Iris-virginica
123           6.3          2.7           4.9          1.8  Iris-virginica
124           6.7          3.3           5.7          2.1  Iris-virginica
125           7.2          3.2           6.0          1.8  Iris-virginica
126           6.2          2.8           4.8          1.8  Iris-virginica
127           6.1          3.0           4.9          1.8  Iris-virginica
128           6.4          2.8           5.6          2.1  Iris-virginica
129           7.2          3.0           5.8          1.6  Iris-virginica
130           7.4          2.8           6.1          1.9  Iris-virginica
131           7.9          3.8           6.4          2.0  Iris-virginica
132           6.4          2.8           5.6          2.2  Iris-virginica
133           6.3          2.8           5.1          1.5  Iris-virginica
134           6.1          2.6           5.6          1.4  Iris-virginica
135           7.7          3.0           6.1          2.3  Iris-virginica
136           6.3          3.4           5.6          2.4  Iris-virginica
137           6.4          3.1           5.5          1.8  Iris-virginica
138           6.0          3.0           4.8          1.8  Iris-virginica
139           6.9          3.1           5.4          2.1  Iris-virginica
140           6.7          3.1           5.6          2.4  Iris-virginica
141           6.9          3.1           5.1          2.3  Iris-virginica
142           5.8          2.7           5.1          1.9  Iris-virginica
143           6.8          3.2           5.9          2.3  Iris-virginica
144           6.7          3.3           5.7          2.5  Iris-virginica
145           6.7          3.0           5.2          2.3  Iris-virginica
146           6.3          2.5           5.0          1.9  Iris-virginica
147           6.5          3.0           5.2          2.0  Iris-virginica
148           6.2          3.4           5.4          2.3  Iris-virginica
149           5.9          3.0           5.1          1.8  Iris-virginica

[150 rows x 5 columns]
>>> print(dataset.describe())
       sepal-length  sepal-width  petal-length  petal-width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.054000      3.758667     1.198667
std        0.828066     0.433594      1.764420     0.763161
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000
>>> print(dataset.groupby('class').size())
class
Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50
dtype: int64
>>> dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
sepal-length       AxesSubplot(0.125,0.53;0.352273x0.35)
sepal-width     AxesSubplot(0.547727,0.53;0.352273x0.35)
petal-length       AxesSubplot(0.125,0.11;0.352273x0.35)
petal-width     AxesSubplot(0.547727,0.11;0.352273x0.35)
dtype: object
>>> plt.show()
>>> dataset.hist()
array([[<matplotlib.axes._subplots.AxesSubplot object at 0x00000000126BC0B8>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x00000000127A1E48>],
       [<matplotlib.axes._subplots.AxesSubplot object at 0x000000001296F5F8>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x0000000016BC2D68>]],
      dtype=object)
>>> plt.show()
>>> scatter_matrix(dataset)
array([[<matplotlib.axes._subplots.AxesSubplot object at 0x0000000016DDD7B8>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x000000001709C320>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x0000000017152A90>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x0000000017196240>],
       [<matplotlib.axes._subplots.AxesSubplot object at 0x00000000171C79B0>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x000000001728D160>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x00000000173648D0>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x00000000173A7080>],
       [<matplotlib.axes._subplots.AxesSubplot object at 0x000000001745D7F0>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x00000000174D5F60>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x0000000017558710>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x0000000017611E80>],
       [<matplotlib.axes._subplots.AxesSubplot object at 0x00000000176A5630>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x00000000176DADA0>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x00000000177A0550>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x00000000177D3CC0>]],
      dtype=object)
>>> plt.show()
>>> array = dataset.values
>>> X = array[:,0:4]
>>> Y = array[:,4]
>>> validation_size = 0.20
>>> seed = 7
>>> X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
>>> # Spot Check Algorithms
>>> models = []
>>> models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
>>> models.append(('LDA', LinearDiscriminantAnalysis()))
>>> models.append(('KNN', KNeighborsClassifier()))
>>> models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
>>> models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
	
>>> # Spot Check Algohrithm
>>> models = []
>>> models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
>>> models.append(('LDR' LinearDiscriminantAnalysis()))
SyntaxError: invalid syntax
>>> models.append(('LDA' LinearDiscriminantAnalysis()))
SyntaxError: invalid syntax
>>> models.append(('LDA', LinearDiscriminantAnalysis()))
>>> models.append(('KNN', KNeighborsClassifier()))
>>> models.append(('LR',LogisticRegression()))
>>> models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
>>> models.append(('CART', DecisionTreeClassifier()))
>>> models.append(('NB', GaussianNB()))
>>> models.append(('SVM', SVC(gamma='auto')))
>>> # evaluate each model in each turn
>>> results = []
>>> names = []
>>> for name,model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	name.append(name)
	msg = "%s: %f (%f)" % (name,cv_results.mean(), cv_results.std())
	print(msg)

	

Traceback (most recent call last):
  File "<pyshell#84>", line 3, in <module>
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
NameError: name 'scoring' is not defined
>>> print(msg)

Traceback (most recent call last):
  File "<pyshell#85>", line 1, in <module>
    print(msg)
NameError: name 'msg' is not defined
>>> for name,model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	name.append(name)
	msg = "%s: %f (%f)" % (name,cv_results.mean(), cv_results.std())
	print(msg)

	

Traceback (most recent call last):
  File "<pyshell#87>", line 3, in <module>
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
NameError: name 'scoring' is not defined
>>> 
>>> 
>>> for name,model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	name.append(name)
	msg = "%s: %f (%f)" % (name,cv_results.mean(), cv_results.std())
	print(msg)


Traceback (most recent call last):
  File "<pyshell#90>", line 3, in <module>
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
NameError: name 'scoring' is not defined
>>> for name,model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	scoring = scoring
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	name.append(name)
	msg = "%s: %f (%f)" % (name,cv_results.mean(), cv_results.std())
	print(msg)

	

Traceback (most recent call last):
  File "<pyshell#92>", line 3, in <module>
    scoring = scoring
NameError: name 'scoring' is not defined
>>> for name,model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold )
	results.append(cv_results)
	name.append(name)
	msg = "%s: %f (%f)" % (name,cv_results.mean(), cv_results.std())
	print(msg)

	

Traceback (most recent call last):
  File "<pyshell#94>", line 5, in <module>
    name.append(name)
AttributeError: 'str' object has no attribute 'append'

>>> for name,model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name,cv_results.mean(), cv_results.std())
	print(msg)

	
LR: 0.966667 (0.040825)
LDA: 0.975000 (0.038188)
KNN: 0.983333 (0.033333)

Warning (from warnings module):
  File "C:\Python27\lib\site-packages\sklearn\linear_model\logistic.py", line 433
    FutureWarning)
FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.

Warning (from warnings module):
  File "C:\Python27\lib\site-packages\sklearn\linear_model\logistic.py", line 460
    "this warning.", FutureWarning)
FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
LR: 0.966667 (0.040825)
LR: 0.966667 (0.040825)
CART: 0.983333 (0.033333)
NB: 0.975000 (0.053359)
SVM: 0.991667 (0.025000)
>>> for name,model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name,cv_results.mean(), cv_results.std())
	print(msg)

	

Traceback (most recent call last):
  File "<pyshell#98>", line 3, in <module>
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
NameError: name 'scoring' is not defined
>>> 
>>> for name,model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name,cv_results.mean(), cv_results.std())
	print(msg)

	
LR: 0.966667 (0.040825)
LDA: 0.975000 (0.038188)
KNN: 0.983333 (0.033333)
LR: 0.966667 (0.040825)
LR: 0.966667 (0.040825)
CART: 0.975000 (0.038188)
NB: 0.975000 (0.053359)
SVM: 0.991667 (0.025000)
>>> # Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
>>> # Compare Algorithm
>>> fig = plt.figure()
>>> fix.subtitle('Algorithm Comparison')

Traceback (most recent call last):
  File "<pyshell#105>", line 1, in <module>
    fix.subtitle('Algorithm Comparison')
NameError: name 'fix' is not defined
>>> fig.subtitle('Algorithm Comparison')

Traceback (most recent call last):
  File "<pyshell#106>", line 1, in <module>
    fig.subtitle('Algorithm Comparison')
AttributeError: 'Figure' object has no attribute 'subtitle'
>>> fix.suptile('Algorithm Comparison')

Traceback (most recent call last):
  File "<pyshell#107>", line 1, in <module>
    fix.suptile('Algorithm Comparison')
NameError: name 'fix' is not defined
>>> fig.suptitle('Algorithm Comparison')
Text(0.5,0.98,'Algorithm Comparison')
>>> ax = fig.add_subplot(111)
>>> plt.boxplot(results)
{'boxes': [<matplotlib.lines.Line2D object at 0x0000000016DB7C50>, <matplotlib.lines.Line2D object at 0x0000000016DAB358>, <matplotlib.lines.Line2D object at 0x0000000016DA0828>, <matplotlib.lines.Line2D object at 0x0000000016CC59B0>, <matplotlib.lines.Line2D object at 0x0000000016D88C88>, <matplotlib.lines.Line2D object at 0x0000000016DF3828>, <matplotlib.lines.Line2D object at 0x000000001263E400>, <matplotlib.lines.Line2D object at 0x000000001248AD30>, <matplotlib.lines.Line2D object at 0x000000001247E0F0>, <matplotlib.lines.Line2D object at 0x0000000012384D68>, <matplotlib.lines.Line2D object at 0x00000000124256A0>, <matplotlib.lines.Line2D object at 0x000000001264AFD0>, <matplotlib.lines.Line2D object at 0x00000000124F1B00>, <matplotlib.lines.Line2D object at 0x0000000012658400>, <matplotlib.lines.Line2D object at 0x0000000011ECA978>, <matplotlib.lines.Line2D object at 0x0000000016EB1D30>, <matplotlib.lines.Line2D object at 0x0000000016FECDD8>], 'fliers': [<matplotlib.lines.Line2D object at 0x0000000016CE37B8>, <matplotlib.lines.Line2D object at 0x0000000016DA0F60>, <matplotlib.lines.Line2D object at 0x0000000016D955C0>, <matplotlib.lines.Line2D object at 0x0000000016D88710>, <matplotlib.lines.Line2D object at 0x0000000016DF32B0>, <matplotlib.lines.Line2D object at 0x000000001263E940>, <matplotlib.lines.Line2D object at 0x000000001248A828>, <matplotlib.lines.Line2D object at 0x000000001247E240>, <matplotlib.lines.Line2D object at 0x00000000124E35C0>, <matplotlib.lines.Line2D object at 0x0000000012425048>, <matplotlib.lines.Line2D object at 0x000000001264A0B8>, <matplotlib.lines.Line2D object at 0x0000000012666908>, <matplotlib.lines.Line2D object at 0x00000000126585F8>, <matplotlib.lines.Line2D object at 0x0000000011ECA320>, <matplotlib.lines.Line2D object at 0x0000000016C627B8>, <matplotlib.lines.Line2D object at 0x0000000016FECA90>, <matplotlib.lines.Line2D object at 0x0000000016C44390>], 'medians': [<matplotlib.lines.Line2D object at 0x0000000016CE3080>, <matplotlib.lines.Line2D object at 0x0000000016EA2048>, <matplotlib.lines.Line2D object at 0x0000000016D95E10>, <matplotlib.lines.Line2D object at 0x0000000016D880F0>, <matplotlib.lines.Line2D object at 0x0000000016C42F60>, <matplotlib.lines.Line2D object at 0x0000000011E34F60>, <matplotlib.lines.Line2D object at 0x000000001248A4E0>, <matplotlib.lines.Line2D object at 0x000000001247EC50>, <matplotlib.lines.Line2D object at 0x00000000124E39E8>, <matplotlib.lines.Line2D object at 0x0000000012425BE0>, <matplotlib.lines.Line2D object at 0x0000000012218390>, <matplotlib.lines.Line2D object at 0x0000000012666550>, <matplotlib.lines.Line2D object at 0x00000000126583C8>, <matplotlib.lines.Line2D object at 0x0000000011ECA7B8>, <matplotlib.lines.Line2D object at 0x0000000016C62080>, <matplotlib.lines.Line2D object at 0x0000000016FEC710>, <matplotlib.lines.Line2D object at 0x0000000016C55FD0>], 'means': [], 'whiskers': [<matplotlib.lines.Line2D object at 0x0000000016DB75C0>, <matplotlib.lines.Line2D object at 0x0000000016DB7A20>, <matplotlib.lines.Line2D object at 0x0000000016DABC50>, <matplotlib.lines.Line2D object at 0x0000000016DABE48>, <matplotlib.lines.Line2D object at 0x0000000016DA0BA8>, <matplotlib.lines.Line2D object at 0x0000000016DA0080>, <matplotlib.lines.Line2D object at 0x0000000016CC5EB8>, <matplotlib.lines.Line2D object at 0x0000000016CC59E8>, <matplotlib.lines.Line2D object at 0x0000000016D889B0>, <matplotlib.lines.Line2D object at 0x0000000016C424A8>, <matplotlib.lines.Line2D object at 0x0000000016DF3CC0>, <matplotlib.lines.Line2D object at 0x0000000016DF3048>, <matplotlib.lines.Line2D object at 0x000000001263E198>, <matplotlib.lines.Line2D object at 0x000000001263E2B0>, <matplotlib.lines.Line2D object at 0x0000000012495B70>, <matplotlib.lines.Line2D object at 0x0000000012495080>, <matplotlib.lines.Line2D object at 0x000000001247E860>, <matplotlib.lines.Line2D object at 0x00000000124E3A58>, <matplotlib.lines.Line2D object at 0x00000000123840F0>, <matplotlib.lines.Line2D object at 0x0000000012384748>, <matplotlib.lines.Line2D object at 0x0000000012425E10>, <matplotlib.lines.Line2D object at 0x0000000012218B00>, <matplotlib.lines.Line2D object at 0x000000001264ABA8>, <matplotlib.lines.Line2D object at 0x000000001264A5C0>, <matplotlib.lines.Line2D object at 0x00000000124F1278>, <matplotlib.lines.Line2D object at 0x00000000124F1AC8>, <matplotlib.lines.Line2D object at 0x00000000126BCB00>, <matplotlib.lines.Line2D object at 0x00000000126BC780>, <matplotlib.lines.Line2D object at 0x0000000011ECAA90>, <matplotlib.lines.Line2D object at 0x0000000017E1B748>, <matplotlib.lines.Line2D object at 0x0000000016EB19B0>, <matplotlib.lines.Line2D object at 0x0000000016EB14E0>, <matplotlib.lines.Line2D object at 0x0000000016C551D0>, <matplotlib.lines.Line2D object at 0x0000000016C55550>], 'caps': [<matplotlib.lines.Line2D object at 0x0000000016CE3D30>, <matplotlib.lines.Line2D object at 0x0000000016CE3E80>, <matplotlib.lines.Line2D object at 0x0000000016DABAC8>, <matplotlib.lines.Line2D object at 0x0000000016DAB0F0>, <matplotlib.lines.Line2D object at 0x0000000016D959E8>, <matplotlib.lines.Line2D object at 0x0000000016D95208>, <matplotlib.lines.Line2D object at 0x0000000016CC5470>, <matplotlib.lines.Line2D object at 0x0000000016CC5320>, <matplotlib.lines.Line2D object at 0x0000000016C421D0>, <matplotlib.lines.Line2D object at 0x0000000016C42E10>, <matplotlib.lines.Line2D object at 0x0000000011E34FD0>, <matplotlib.lines.Line2D object at 0x0000000011E34470>, <matplotlib.lines.Line2D object at 0x000000001248ADD8>, <matplotlib.lines.Line2D object at 0x000000001248A7F0>, <matplotlib.lines.Line2D object at 0x0000000012495390>, <matplotlib.lines.Line2D object at 0x00000000124959E8>, <matplotlib.lines.Line2D object at 0x00000000124E3828>, <matplotlib.lines.Line2D object at 0x00000000124E3F28>, <matplotlib.lines.Line2D object at 0x0000000012384EB8>, <matplotlib.lines.Line2D object at 0x0000000012425C18>, <matplotlib.lines.Line2D object at 0x0000000012218278>, <matplotlib.lines.Line2D object at 0x0000000012218A20>, <matplotlib.lines.Line2D object at 0x0000000012666EB8>, <matplotlib.lines.Line2D object at 0x0000000012666630>, <matplotlib.lines.Line2D object at 0x00000000124F1DD8>, <matplotlib.lines.Line2D object at 0x0000000012658C50>, <matplotlib.lines.Line2D object at 0x00000000126BC8D0>, <matplotlib.lines.Line2D object at 0x00000000126BCF28>, <matplotlib.lines.Line2D object at 0x0000000016C62D30>, <matplotlib.lines.Line2D object at 0x0000000016C62EF0>, <matplotlib.lines.Line2D object at 0x0000000016EB1080>, <matplotlib.lines.Line2D object at 0x0000000016FEC390>, <matplotlib.lines.Line2D object at 0x0000000016C558D0>, <matplotlib.lines.Line2D object at 0x0000000016C55C50>]}
ax
>>> ax.set_xticklabels(names)
[Text(0,0,'LR'), Text(0,0,'LDA'), Text(0,0,'KNN'), Text(0,0,'LR'), Text(0,0,'LR'), Text(0,0,'CART'), Text(0,0,'NB'), Text(0,0,'SVM'), Text(0,0,'LR'), Text(0,0,'LDA'), Text(0,0,'KNN'), Text(0,0,'LR'), Text(0,0,'LR'), Text(0,0,'CART'), Text(0,0,'NB'), Text(0,0,'SVM')]
>>> plt.show()
>>> 
