# Codigo escrito en Spyder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importar el data set
df = pd.read_csv('adult.data', header= None)

df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
              'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
              'native-country', '>50K_<=50K']
    # I could not import the UCI dataset with the correct column labels, so I had to place them manually.

# EDA
print(df.shape)

features = df.columns.values
for f in features :
    print(f,': ',df[f].unique())
    print()
    # Workclas, occupation, native-country contain a value " ?" which should be treated as a null value.
        
print(df.info)

df_object = df.select_dtypes(include=[object])
for f in df_object:
    print(df[f].value_counts())
df_object = df_object.iloc[:, :-1]

print(df.duplicated().sum())
df = df.drop_duplicates()
print(df.shape)


# Change the value " ?" to null value
for i in df.columns:
    df[i].replace(' ?', np.nan, inplace=True)
print(df.isna().sum())

print(df.isnull().mean().sort_values(ascending=False))
    # The amount of null data is not very large, it would not be bad to delete them
import missingno as msno
msno.matrix(df)
#   Let's see if there are any correlations between our null data
msno.heatmap(df)
msno.matrix(df.sort_values("workclass"))
    # When we do not have information about whether an individual has an occupation,
    # we cannot have information about his or her type of occupation either.

descripction = df.describe().T


bins=[0,30,50,80]
sns.countplot(x=pd.cut(df.age, bins=bins), hue=df[">50K_<=50K"])
plt.show() 
# Between the ages of 30 and 50 there are more people earning more than 50k.

sns.countplot(x=df.sex, hue=df[">50K_<=50K"])
plt.show() # Male: 21790, Female: 10771
# 50% Of men earn more than 50k per month, but on the other hand, only 20% of women earn
# more than 50k per month.

bins2 = [0,20,40,60,80,100]
sns.countplot(x=pd.cut(df['hours-per-week'], bins=bins2), hue=df["sex"])
plt.show() 
# From the age of 20, men dedicate more hours to work than women; the difference is
# greater from the age of 40 onwards. 

plt.figure(figsize=(10,7))
sns.countplot(x=df.workclass, hue=df[">50K_<=50K"])
plt.show()

plt.figure(figsize=(16,10))
sns.countplot(x=df.education, hue=df[">50K_<=50K"])
plt.show() 
# If you have a master's degree, you are more likely to earn more than 50 thousand.

plt.figure(figsize=(10,7))
sns.lineplot(x=df["education-num"], y =df[">50K_<=50K"])
plt.title('education num')
plt.show() 
# The wage gap accelerates more rapidly after 12 years of education. The peak is reached at 16 years of age

plt.figure(figsize=(10,7))
sns.distplot(x=df["education-num"])
plt.title('Education num')
plt.show()

plt.figure(figsize=(10,7))
sns.lineplot(x=df["hours-per-week"], y =df[">50K_<=50K"])
plt.title('hours-per-week')
plt.show() 
# Between 60 and 70 hours per week are most people earning over 50k, if you work 
# more or less than that you are less likely to exceed 50k.

plt.figure(figsize=(10,7))
sns.distplot(x=df["hours-per-week"])
plt.title('Hours per week')
plt.show() # Most people work about 40 hours per week

df_number = df.select_dtypes(include=[np.number])
def plot_uni(d):
    f,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,5))
    sns.histplot(d, kde=True, ax=ax[0])
    ax[0].axvline(d.mean(), color='y', linestyle='--',linewidth=2)
    ax[0].axvline(d.median(), color='r', linestyle='dashed', linewidth=2)
    ax[0].axvline(d.mode()[0],color='g',linestyle='solid',linewidth=2)
    ax[0].legend({'Mean':d.mean(),'Median':d.median(),'Mode':d.mode()})
    
    sns.boxplot(x=d, showmeans=True, ax=ax[1])
    plt.tight_layout()
for f in df_number:
    plot_uni(df[f])
# Age in the dataset could have a more Gaussian distribution
# we must work on the variance of fnlwgt and hours-per-week

df.dropna(inplace=True)
print(df.isna().sum())


# Cleaning outliers with IQR
df_eda = df
outliers_cols = []
for col in df_number.columns:
    q75,q25 = np.percentile(df_number.loc[:,col],[75,25])
    iqr = q75-q25
     
    max = q75+(1.5*iqr)
    min = q25-(1.5*iqr)
    if any(df_number[col].values < min) or any(df_number[col].values > max):
        outliers_cols.append(col)
outliers_cols = ['age', 'fnlwgt']

for col in df_number:
    if col in outliers_cols:
        q75,q25 = np.percentile(df_eda.loc[:,col],[75,25])
        iqr = q75-q25
     
        max = q75+(1.5*iqr)
        min = q25-(1.5*iqr)
     
        df_eda.loc[df[col] < min,col] = np.nan
        df_eda.loc[df[col] > max,col] = np.nan
print(df_eda.isna().sum())
df_eda.dropna(inplace=True)

# Transforming the Variables to attain Normal Distribution
from sklearn import preprocessing
pt = preprocessing.PowerTransformer()
for col in df_number.columns:
    df_eda[col] = pt.fit_transform(df_eda[col].values.reshape(-1,1))

    #Graph    
for col in df_number:
  f,ax = plt.subplots(nrows=1,ncols=1,figsize=(12,8))    
  sns.kdeplot(data = df_eda, x = col, hue = '>50K_<=50K', fill = 'dark', palette = 'dark' )
  plt.tight_layout() 

X = df_eda.iloc[:, :-1]
y = df_eda.iloc[:,-1]

# Coding categorical data
from sklearn import preprocessing
le_x = preprocessing.LabelEncoder()
y = le_x.fit_transform(y)
for n in df_object:
    X[n] = le_x.fit_transform(X[n])
    
# Imbalance 
from imblearn.combine import SMOTETomek
smk = SMOTETomek(random_state = 15)
print(df_eda['>50K_<=50K'].value_counts())  
X_res,y_res = smk.fit_resample(X,y)

X = X_res
y = y_res

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Models
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
models = pd.DataFrame(columns=["Model","Accuracy Score"])

# -------------------- Logistic Regression --------------------
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state = 0)
log_reg.fit(X_train, y_train)
predictions = log_reg.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap = plt.cm.Blues)
plt.title('Logistic Regression')
print(classification_report(y_test, predictions))

score = accuracy_score(y_test, predictions)

new_row = {"Model": "LogisticRegression", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

# -------------------- K-Nearest Neighbors (Knn) --------------------
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap = plt.cm.Blues)
plt.title('KNN')
print(classification_report(y_test, predictions))

score = accuracy_score(y_test, predictions)

new_row = {"Model": "KNN", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

# -------------------- Suport Vector Machine (svm) --------------------
from sklearn.svm import SVC
svm = SVC(kernel = "rbf", random_state = 0)
svm.fit(X_train, y_train) 
predictions = svm.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap = plt.cm.Blues)
plt.title('SVM')
report = (classification_report(y_test, predictions))

score = accuracy_score(y_test, predictions)

new_row = {"Model": "SVM", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

# -------------------- Naive bayes --------------------
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap = plt.cm.Blues)
plt.title('Naive Bayes')
report = (classification_report(y_test, predictions))

score = accuracy_score(y_test, predictions)

new_row = {"Model": "NaiveBayes", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

# -------------------- Random Forest --------------------
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(n_estimators = 200, criterion = "gini", random_state = 0)
randomforest.fit(X_train, y_train)
predictions = randomforest.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap = plt.cm.Blues)
plt.title('Random Forest')
report = (classification_report(y_test, predictions))

score = accuracy_score(y_test, predictions)

new_row = {"Model": "RandomForest", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

# -------------------- XGBoost --------------------
from xgboost import XGBClassifier
XGB = XGBClassifier()
XGB.fit(X_train, y_train)
predictions = XGB.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap = plt.cm.Blues)
plt.title('XGBoost')
report = (classification_report(y_test, predictions))

score = accuracy_score(y_test, predictions)

new_row = {"Model": "XGBoost", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

# -------------------- CatBoost --------------------
from catboost import CatBoostClassifier
CatBoost = CatBoostClassifier(verbose=False)
CatBoost.fit(X_train,y_train,eval_set=(X_test, y_test))
predictions = CatBoost.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap = plt.cm.Blues)
plt.title('CatBoost')
report = (classification_report(y_test, predictions))

score = accuracy_score(y_test, predictions)

new_row = {"Model": "CatBoost", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

# -------------------- ExtraTreeClassifier --------------------
from sklearn.ensemble import ExtraTreesClassifier
etc = ExtraTreesClassifier(n_estimators = 200,
                                        criterion ='entropy', max_features = 'auto')
etc.fit(X_train,y_train)
predictions = etc.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap = plt.cm.Blues)
plt.title('ExtraTreeClassifier')
report = (classification_report(y_test, predictions))

score = accuracy_score(y_test, predictions)

new_row = {"Model": "ETC", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

# -------------------- GradientBoostingClassifier --------------------
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
predictions = gbc.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap = plt.cm.Blues)
plt.title('GradientBoostingClassifier')
report = (classification_report(y_test, predictions))

score = accuracy_score(y_test, predictions)

new_row = {"Model": "GBC", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

models = models.sort_values(by="Accuracy Score", ascending=False)

# Cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = CatBoost, X = X_train, y = y_train, cv = 10)
print(accuracies.mean()) # Sesgo
print(accuracies.std()) # Varianza

tmp=pd.DataFrame({'feature':X.columns,
                 'importance':CatBoost.feature_importances_}).sort_values(by='importance',ascending=False)
plt.figure(figsize=(10,8))
sns.barplot(x=tmp.importance ,y=tmp.feature).set_title('Feature Importance')
plt.show()

# Hyperparameter tuning With Optuna
import optuna  
from sklearn.metrics import recall_score
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_slice

# XGBoost
def objetive(trial):
    param = {
            'booster': trial.suggest_categorical("booster", ['gbtree', 'gblinear', 'dart']),
            'lambda': trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "n_estimators": trial.suggest_int("n_estimatos", 500, 1000),
            "gamma": trial.suggest_int("gamma", 0, 2),
            "max_depth": trial.suggest_int("max_dept", 3, 10),
            "n_jobs": (-1)
        }

    trial_xgb = XGBClassifier(**param, random_state = 1)
    predictions = trial_xgb.predict(X_test)
    trial_xgb.fit(X_train,y_train)

    score = recall_score(y_test, predictions)
    score1 = accuracy_score(y_test, predictions)
    print("---. ","Recall: ",score," .---")
    print("---. ","Accuracy: ",score1," .---")
    return score

study = optuna.create_study(direction = 'maximize')
study.optimize(objetive, n_trials = 50, n_jobs= -1)

study.best_params
xgb_tuned = study.best_params

plot_optimization_history(study)
plot_param_importances(study)
plot_slice(study, ['n_estimators','max_depth'])

# CatBoost
def objetive1(trial):
    param1 = {
        'depth': trial.suggest_int('depth', 3, 10),
          'iterations': trial.suggest_int('iterations', 250, 1000),
          'learning_rate': trial.suggest_float('learning_rate', 0.03 ,0.3), 
          'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 3, 100),
          'border_count': trial.suggest_int('border_count', 32, 200)
          }
    trial_cat = CatBoostClassifier(**param1, random_state = 1)
    trial_cat.fit(X_train,y_train)
    predictions = trial_cat.predict(X_test)
    

    score = recall_score(y_test, predictions)
    score1 = accuracy_score(y_test, predictions)
    print("---. ","Recall: ",score," .---")
    print("---. ","Accuracy: ",score1," .---")
    return score1

study1 = optuna.create_study(direction = 'maximize')
study1.optimize(objetive1, n_trials = 50, n_jobs= -1)

study1.best_params
cat_tuned = study1.best_params

plot_optimization_history(study1)
plot_param_importances(study1)
plot_slice(study1, ['n_estimators','max_depth'])

# Extra Tree Classifier
def objetive2(trial):
    param2 = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
          'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
          'max_depth': trial.suggest_int('max_depth', 3 ,10), 
          'max_features': trial.suggest_float('max_features', 0.25, 1.0)
          }
    trial_etc = ExtraTreesClassifier(**param2, random_state = 1)
    trial_etc.fit(X_train,y_train)
    predictions = trial_etc.predict(X_test)
    

    score = recall_score(y_test, predictions)
    score1 = accuracy_score(y_test, predictions)
    print("---. ","Recall: ",score," .---")
    print("---. ","Accuracy: ",score1," .---")
    return score1

study2 = optuna.create_study(direction = 'maximize')
study2.optimize(objetive2, n_trials = 70, n_jobs= -1)

study2.best_params
etc_tuned = study2.best_params

plot_optimization_history(study2)
plot_param_importances(study2)
plot_slice(study2, ['n_estimators','max_depth'])

# Random Forest
def objetive3(trial):
    param3 = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
          'bootstrap': trial.suggest_categorical('boostrap', ['True', 'False']),
          'max_depth': trial.suggest_int('max_depth', 3 ,10), 
          'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt']),
          'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
          'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
          }
    trial_rf = RandomForestClassifier(**param3, random_state = 1)
    trial_rf.fit(X_train,y_train)
    predictions = trial_rf.predict(X_test)
    

    score = recall_score(y_test, predictions)
    score1 = accuracy_score(y_test, predictions)
    print("---. ","Recall: ",score," .---")
    print("---. ","Accuracy: ",score1," .---")
    return score1

study3 = optuna.create_study(direction = 'maximize')
study3.optimize(objetive3, n_trials = 70, n_jobs= -1)

study3.best_params
rf_tuned = study3.best_params

plot_optimization_history(study3)
plot_param_importances(study3)
plot_slice(study3, ['n_estimators','max_depth'])

# Gradient Boosting Classifier
def objetive4(trial):
    param4 = {
          'loss': trial.suggest_categorical('loss', ['deviance', 'exponential']),
          'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
          'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.1),
          'max_depth': trial.suggest_int('max_depth', 3 ,10), 
          'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt']),
          'criterion': trial.suggest_categorical('criterion', ['friedman_mse', 'squared_error'])
          }
    trial_gbc = GradientBoostingClassifier(**param4, random_state = 1)
    trial_gbc.fit(X_train,y_train)
    predictions = trial_gbc.predict(X_test)
    

    score = recall_score(y_test, predictions)
    score1 = accuracy_score(y_test, predictions)
    print("---. ","Recall: ",score," .---")
    print("---. ","Accuracy: ",score1," .---")
    return score1

study4 = optuna.create_study(direction = 'maximize')
study4.optimize(objetive4, n_trials = 150, n_jobs= -1)

study4.best_params
gbc_tuned = study4.best_params

plot_optimization_history(study4)
plot_intermediate_values(study4)
plt.figure(figsize = (12,8))
plot_contour(study4)
plot_param_importances(study4)
plot_slice(study4)
plot_edf(study4)
plot_slice(study4, ['n_estimators','max_depth'])

# Voting Classifier
from sklearn.ensemble import VotingClassifier
eclf1 = VotingClassifier(estimators= [('ExtraTreeClassifier', ExtraTreesClassifier(**etc_tuned)), 
                                      ('CatBoost', CatBoost), ('Randomforest', randomforest), 
                                      ('XGBoost', XGB),
                                      ('Gradient Boosting', GradientBoostingClassifier(**gbc_tuned))])

eclf1.fit(X_train, y_train)
voting_pred= eclf1.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, voting_pred, cmap = plt.cm.Blues)
plt.title('voting_pred')

print(classification_report(y_test, voting_pred))
