import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importar el data set
dataset = pd.read_csv('adult.data')
final_test = pd.read_csv('adult.test')

# ANalysis
sns.heatmap(data=dataset.corr(),cmap="YlGnBu", annot=True ,linewidths=0.2, linecolor='white')
features = dataset.columns.values
for f in features :
    print(f,': ',dataset[f].unique())
    print()
# ? value in State-gov implica que hay valores desconocidos 
# ? value in Adm-clerical implica que hay valores desconocidos 

print(dataset.isna().sum())
# Los valores nulos estan representados como '?' en solo 2 columnas categoricas

descripction = dataset.describe().T

categoric_descrip = dataset.describe(include=['O'])

categoric_col = dataset.select_dtypes('object').columns.values

# Separar datos categoricos y datos numericos
df_object = dataset.select_dtypes(include=[object])
df_number = dataset.select_dtypes(include=[np.number])

# print(df_object.White.value_counts())

# Limpieza de NAs
for i in dataset.columns:
    dataset[i].replace(' ?', np.nan, inplace=True)
dataset.dropna(inplace=True)

for f in features :
    print(f,': ',dataset[f].unique())
    print()
print(dataset.duplicated().sum())


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Codificar datos categoricos
from sklearn import preprocessing
le_x = preprocessing.LabelEncoder()
X[:, 1] = le_x.fit_transform(X[:, 1])
X[:, 2] = le_x.fit_transform(X[:, 2])
X[:, 3] = le_x.fit_transform(X[:, 3])
X[:, 5] = le_x.fit_transform(X[:, 5])
X[:, 6] = le_x.fit_transform(X[:, 6])
X[:, 7] = le_x.fit_transform(X[:, 7])
X[:, 8] = le_x.fit_transform(X[:, 8])
X[:, 9] = le_x.fit_transform(X[:, 9])
X[:, 13] = le_x.fit_transform(X[:, 13])
y = le_x.fit_transform(y)


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Models
from sklearn import metrics
from sklearn.metrics import accuracy_score
models = pd.DataFrame(columns=["Model","Accuracy Score"])

# -------------------- Logistic Regression --------------------
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state = 0)
log_reg.fit(X_train, y_train)
predictions = log_reg.predict(X_test)

score = accuracy_score(y_test, predictions)
print("LogisticRegression: ", score)

new_row = {"Model": "LogisticRegression", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

# -------------------- K-Nearest Neighbors (Knn) --------------------
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

score = accuracy_score(y_test, predictions)
print("KNN: ", score)

new_row = {"Model": "KNN", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

# -------------------- Suport Vector Machine (svm) --------------------
from sklearn.svm import SVC
svm = SVC(kernel = "rbf", random_state = 0)
svm.fit(X_train, y_train) 
predictions = svm.predict(X_test)

score = accuracy_score(y_test, predictions)
print("SVM: ", score)

new_row = {"Model": "SVM", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

# -------------------- Kernel SVM --------------------
from sklearn.svm import SVC
k_svm = SVC(kernel = "rbf", random_state = 0)
k_svm.fit(X_train, y_train)
predictions = k_svm.predict(X_test)

score = accuracy_score(y_test, predictions)
print("Kernel SVM: ", score)

new_row = {"Model": "KernelSVM", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

# -------------------- Naive bayes --------------------
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

score = accuracy_score(y_test, predictions)
print("NaiveBayes: ", score)

new_row = {"Model": "NaiveBayes", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

# -------------------- Decision Tree --------------------
from sklearn.tree import DecisionTreeClassifier
d_tree = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
d_tree.fit(X_train, y_train)
predictions = d_tree.predict(X_test)

score = accuracy_score(y_test, predictions)
print("DecisionTree: ", score)

new_row = {"Model": "DecisionTree", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

# -------------------- Random Forest --------------------
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(n_estimators = 10, criterion = "gini", random_state = 0)
randomforest.fit(X_train, y_train)
predictions = randomforest.predict(X_test)

score = accuracy_score(y_test, predictions)
print("Random Forest: ", score)

new_row = {"Model": "RandomForest", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)


models.sort_values(by="Accuracy Score", ascending=False)



# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)

# Best model in this DataSet: Random Forest Accuracy 0.845 