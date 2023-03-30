# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import joblib

# %%
rainfall = pd.read_csv("C:/Users/chand/OneDrive/Desktop/codegood/cpp/demo/hello/rainfall/datarainfall.csv")
pd.set_option("display.max_columns", None)
rainfall

# %%
rainfall.head()

# %%
rainfall.shape

# %%
rainfall.info()

# %%
rainfall.describe()

# %%
rainfall.hist(bins=50, figsize=(12, 8))
plt.show()


# %%
def shuffle_and_split_data(data, test_ratio):
 shuffled_indices = np.random.permutation(len(data))
 test_set_size = int(len(data) * test_ratio)
 test_indices = shuffled_indices[:test_set_size]
 train_indices = shuffled_indices[test_set_size:]
 return data.iloc[train_indices], data.iloc[test_indices]

# %%
train_set, test_set = shuffle_and_split_data(rainfall, 0.2)

# %%
len(test_set)

# %%
len(train_set)

# %%
corrmat = rainfall.corr(method = "spearman")
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(corrmat,annot=True)

# %%
features = list(rainfall.select_dtypes(include = np.number).columns)
features.remove('DY')
features.remove('MO')
features.remove('YEAR')
print(features)

# %%
plt.subplots(figsize=(15,8))
 
for i, col in enumerate(features):
  plt.subplot(3,4, i + 1)
  sns.boxplot(rainfall[col])
plt.tight_layout()
plt.show()

# %%
target = rainfall.RAIN

# %%
plt.pie(rainfall['RAIN'].value_counts().values,
        labels = rainfall['RAIN'].value_counts().index,
        autopct='%1.1f%%')
plt.show()

# %%
rainfall.replace({'YES':1, 'NO':0}, inplace=True)

# %%
target = rainfall.RAIN

# %%
features = rainfall.drop(['DY', 'RAIN', 'PRECTOTCORR' , 'YEAR'], axis=1)
target = rainfall.RAIN

# %%
X_train, X_val, Y_train, Y_val = train_test_split(features,
                                                  target,
                                                  test_size = 0.2,
                                                  stratify = target,
                                                  random_state=2)
ros = RandomOverSampler(sampling_strategy='minority',
                        random_state=22)
X, Y = ros.fit_resample(X_train,Y_train)

# %%
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)

# %%
models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf', probability=True)]
 
for i in range(3):
  models[i].fit(X, Y)
 
  print(f'{models[i]} : ')
 
  train_preds = models[i].predict_proba(X)
  print('Training Accuracy : ', metrics.roc_auc_score(Y, train_preds[:,1]))
 
  val_preds = models[i].predict_proba(X_val)
  print('Validation Accuracy : ', metrics.roc_auc_score(Y_val, val_preds[:,1]))
  print()

# %%

print(models[2].predict([[1,20.87,12.29,19.74,28.66,13.1,8.85,101.12,1.47,254.75,2.01,253.88]]))

# %%

y_pred = models[2].predict(X_val)
print(confusion_matrix(Y_val,y_pred))


