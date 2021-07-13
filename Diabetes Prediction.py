#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib .pyplot as plt
import numpy as np
import sklearn as skl


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing


# In[3]:


column_name = ['pregnant','glucose','Blood Pressure','skin_Thickness','Insulin','BMI','pedigree','Age','Label']
data= pd.read_csv ("diabetes.csv",header=None, names=column_name, skiprows=(0,0))  
#skiprow skips the first row


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


data.describe()                         #mean/standard_deviation etc for every paramenter


# In[7]:


#converting string to numeric value
convert_col = ['pregnant','glucose','Blood Pressure','skin_Thickness','Insulin','BMI','pedigree','Age']

for col in convert_col :
    data[col] =pd.to_numeric(data[col])


# In[8]:


feature_cols=['pregnant','glucose','Blood Pressure','skin_Thickness','Insulin','BMI','pedigree','Age']
X=data[feature_cols]
y=data.Label


# In[9]:


#CORRELATION

corr=data.corr()
plt.figure(figsize=(100,100))

coor_range =corr[(corr >=0.3) | (corr<= -0.1)] 


# In[10]:


sns.heatmap(coor_range ,vmax=0.8 ,linewidths =0.01,square =True, annot =True, cmap='GnBu', linecolor="white", cbar_kws={'label':'Feature Correlation Color'})
plt.ylabel("feature values on Y axis")
plt.xlabel("Feature Values On X Axis")
plt.title('Correlation between Features Of The Dataset')


# In[11]:


#splitting data
X_train,X_test,y_train,y_test =train_test_split (X,y, test_size=0.20, random_state=42)


# In[12]:


#logistic regression algorithm

Log_func = LogisticRegression()


# In[13]:


#fitting the model with the training data

Log_func.fit(X_train,y_train)
y_prediction = Log_func.predict(X_test)


# In[14]:


#model evaluation using Confusion Matrix for the peformance of classification model

from sklearn import metrics

cnf_matrix_evaluation =metrics.confusion_matrix(y_test,y_prediction)
cnf_matrix_evaluation


# In[15]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


class_names =[0,1] #naming the classes

fig , ax =plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)

#making the heatmap for getting the confusion matrix

sns.heatmap(pd.DataFrame(cnf_matrix_evaluation),annot=True, cmap="YlGnBu",fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion Matrix : Diabetes Patient' , y=1.1)
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")


# In[17]:


#confusion matrix conclusion for evaluation metrics

print("ACCURACY: ",metrics.accuracy_score(y_test,y_prediction))
print("PRECISION: ",metrics.precision_score(y_test,y_prediction))
print("RECALL: ",metrics.recall_score(y_test,y_prediction))
print(metrics.accuracy_score(y_test,y_prediction)*100 , "% : ", "chances that the person heving diabetes in the present dataset")

print(y_prediction)


# In[19]:


#Performanxe evaluation using the ROC curve

y_prediction_probability = Log_func.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_prediction_probability)
auc = metrics.roc_auc_score(y_test, y_prediction_probability)

plt.plot(fpr,tpr,label="DATA 1 , auc="+str(auc))
plt.legend()

plt.show()


# In[22]:


from sklearn.metrics import f1_score


# In[28]:


#F1_Score
f1_score(y_test,y_prediction, average=None)


# In[ ]:




