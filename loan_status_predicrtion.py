

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score

"""Data Collection and processing"""

data=pd.read_csv('/content/loan_data.csv')

data.head()

data.shape

data.isnull().sum()

data=data.dropna()

data.isnull().sum()

"""Data Visualization"""

sns.countplot(x='Education', hue='Loan_Status', data=data)

sns.countplot(x='Married', hue='Loan_Status', data=data)

"""Convert categorical values to numerical values"""

data.replace({'Loan_Status':{'N':0,'Y':1}}, inplace=True)

data['Dependents'].value_counts()

data.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)

data.head(10)

#Diffrentiate data in feature and target
features=data.drop(columns=['Loan_ID','Loan_Status','Dependents'],axis=1)
target=data[['Loan_Status']]

x_train,x_test,y_train,y_test=train_test_split(features,target,train_size=0.8,random_state=1)

#training the model
model=svm.SVC(kernel='linear')
model.fit(x_train,y_train)

prediction=model.predict(x_test)

accuracy_score(prediction,y_test)

lmodel=LogisticRegression()

lmodel.fit(x_train,y_train)

p=lmodel.predict(x_test)

accuracy_score(p,y_test)

print('Accuracy score of test data ',accuracy_score(p,y_test))

"""Making a predictive system"""

input_data=(1,1,1,0,4583,1508.0,128.0,360.0,1.0,0,0)
input_data_np=np.asarray(input_data)
input_data_reshape=input_data_np.reshape(1,-1)
p=model.predict(input_data_reshape)
if[p=='Y']:
  print('Your loan will approved')
else:
    print('Your loan will not approved')
