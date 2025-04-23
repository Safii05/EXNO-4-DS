# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
```
```
df=pd.read_csv("/content/bmi.csv")
```
```
df.head()
```
![image](https://github.com/user-attachments/assets/2727c53b-8068-4a57-b0b0-e7274e3fe6db)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/edab8eb6-944c-46e6-84c4-8514cd7c7223)
```
max_vals = df[['Height', 'Weight']].abs().max()
print(max_vals)
```
![image](https://github.com/user-attachments/assets/f17626fb-c95e-491a-b8ff-15d20580286e)
```
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df[['Height','Weight']] = sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/84aa1f8a-1141-4a6a-9420-d958f73971d4)
```
from sklearn.preprocessing import MinMaxScaler
```
```
scaler = MinMaxScaler()
df[['Height','Weight']] = scaler.fit_transform(df[['Height','Weight']])
```
```
df.head(10)
```
![image](https://github.com/user-attachments/assets/44af8e40-56a5-46c9-ac27-4a73679206cc)
```
from sklearn.preprocessing import Normalizer
scaler = Normalizer()
df[['Height','Weight']] = scaler.fit_transform(df[['Height','Weight']])
```
```
df
```
![image](https://github.com/user-attachments/assets/01ad01d7-58d8-4579-b449-6e8e4a497bba)
```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/4538b410-d9e4-4322-92af-67cbf550c9c6)
```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/0a384e5e-0160-40c4-93c1-4c6e94dbdf11)
```
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/3f380d62-d904-480d-9a7c-e47c2fe0ee3c)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/user-attachments/assets/35dc5c46-ea2b-4d17-aa16-2d5a0561c2b1)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-squared statistic: {chi2}")
print(f"P-value: {p}")
```
![image](https://github.com/user-attachments/assets/a0236beb-fa1e-461d-965f-56684b7bae95)
```
from sklearn.feature_selection import SelectKBest,mutual_info_classif,f_classif
data={
'Feature1':[1,2,3,4,5],
'Feature2':['A','B','C','A','B'],
'Feature3':[0,1,1,0,1],
'Target':[0,1,1,0,1]
}
df=pd.DataFrame(data)
X=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
X_new=selector.fit_transform(X,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/45e4e7cf-dbc9-42d8-b9a3-af03d4554134)
```


# RESULT:
  Feature scaling and feature selection process has been successfullyperformed on the
data set.

