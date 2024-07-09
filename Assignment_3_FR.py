#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# ## Data Collection

# In[2]:


data = pd.read_csv('winequality-red.csv')


# In[3]:


data


# In[4]:


data.head()


# In[5]:


data.isnull().sum()  #checking for missing values


# In[6]:


data.shape #data size


# ## Data Analysis and Visulaization

# In[7]:


data.describe() #statistical measures of our data


# In[8]:


sns.catplot(x='quality', data=data, kind='count') #number of values for each quality


# In[9]:


value_counts_quality = data['quality'].value_counts()
print(value_counts_quality)


# In[10]:


#volatitle aciditiy vs quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='volatile acidity',data=data)


# In[11]:


#ctric acid contain vs quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='citric acid',data=data)


# In[12]:


correlation = data.corr()


# In[13]:


correlation


# In[14]:


print(correlation)


# In[15]:


#constructing heatmap for better understanding of correlation ( between different parameters and quality)
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt = '.1f', annot = True, annot_kws={'size':6}, cmap = 'Greens')


# ## Data Preprocessing

# In[16]:


#seperating data and variable
X= data.drop('quality',axis=1)
X


# In[17]:


Y = data['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)
Y


# ## Train and Test Split

# In[18]:


X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2,random_state=2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_test_scaled)
print(X_train_scaled)


# In[19]:


print(X_train)
print(X_test)
print(Y_train)
print(Y_test)


# In[20]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


print(X.shape)
print(Y.shape)


# ## Model Training

# In[21]:


#Random Forest Classifier
model = RandomForestClassifier()


# In[22]:


# #accuracy on test data
# X_test_predicition = model.predict(X_test)
# test_data_accuracy = accuracy_score(X_test_predicition, Y_test)
# test_data_accuracy


# accuracy = accuracy_score(Y_test, X_test_predicition1)
# precision = precision_score(Y_test, X_test_predicition1, average='weighted')
# recall = recall_score(Y_test, X_test_predicition1, average='weighted')
# f1 = f1_score(Y_test, X_test_predicition1, average='weighted')

# print(f'Accuracy: {accuracy:.2f}')
# print(f'Precision: {precision:.2f}')
# print(f'Recall: {recall:.2f}')
# print(f'F1 Score: {f1:.2f}')

# cv_scores = cross_val_score(model, X_train, Y_train, cv=5)
# cv_scores


# In[23]:


#Desicion Tree
model1 = DecisionTreeClassifier()
model1.fit(X_train, Y_train)


# In[24]:


#accuracy on test data
X_test_predicition2 = model1.predict(X_test)
test_data_accuracy = accuracy_score(X_test_predicition2, Y_test)
test_data_accuracy


accuracy = accuracy_score(Y_test, X_test_predicition2)
precision = precision_score(Y_test, X_test_predicition2, average='weighted')
recall = recall_score(Y_test, X_test_predicition2, average='weighted')
f1 = f1_score(Y_test, X_test_predicition2, average='weighted')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

cv_scores = cross_val_score(model1, X_train, Y_train, cv=5)
cv_scores


# In[25]:


#SVM
model2 = SVC()
model2.fit(X_train, Y_train)


# In[26]:


#accuracy on test data
X_test_predicition = model2.predict(X_test)
test_data_accuracy = accuracy_score(X_test_predicition, Y_test)
test_data_accuracy

accuracy = accuracy_score(Y_test, X_test_predicition)
precision = precision_score(Y_test, X_test_predicition, average='weighted')
recall = recall_score(Y_test, X_test_predicition, average='weighted')
f1 = f1_score(Y_test, X_test_predicition, average='weighted')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

cv_scores = cross_val_score(model2, X_train, Y_train, cv=5)
cv_scores


# In[27]:


#KNN
model3 = KNeighborsClassifier()
model3.fit(X_train, Y_train)


# In[28]:


#accuracy on test data
X_test_predicition = model3.predict(X_test)
test_data_accuracy = accuracy_score(X_test_predicition, Y_test)
test_data_accuracy

accuracy = accuracy_score(Y_test, X_test_predicition)
precision = precision_score(Y_test, X_test_predicition, average='weighted')
recall = recall_score(Y_test, X_test_predicition, average='weighted')
f1 = f1_score(Y_test, X_test_predicition, average='weighted')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

cv_scores = cross_val_score(model3, X_train, Y_train, cv=5)
cv_scores


# ## Model Performance Summary:
# Random Forest:
# 
# Accuracy: 0.91
# Precision: 0.91
# Recall: 0.91
# F1 Score: 0.91
# Cross-Validation Scores: [0.8828125, 0.921875, 0.8984375, 0.890625, 0.91372549]
# Decision Tree:
# 
# Accuracy: 0.87
# Precision: 0.88
# Recall: 0.87
# F1 Score: 0.87
# Cross-Validation Scores: [0.8125, 0.8359375, 0.84375, 0.875, 0.88235294]
# SVM:
# 
# Accuracy: 0.88
# Precision: 0.89
# Recall: 0.88
# F1 Score: 0.82
# Cross-Validation Scores: [0.86328125, 0.86328125, 0.86328125, 0.859375, 0.8627451]
# KNN:
# 
# Accuracy: 0.89
# Precision: 0.88
# Recall: 0.89
# F1 Score: 0.88
# Cross-Validation Scores: [0.84765625, 0.8359375, 0.84765625, 0.84375, 0.86666667]
# Analysis and Best Model Selection:
# Accuracy: Random Forest has the highest accuracy (0.91), indicating that it correctly predicts the labels for 91% of the instances.
# Precision: Random Forest also has the highest precision (0.91), meaning it has a lower false positive rate compared to the other models.
# Recall: Random Forest again has the highest recall (0.91), meaning it correctly identifies 91% of the actual positives.
# F1 Score: The F1 score of Random Forest is also the highest (0.91), which is a balance between precision and recall.
# Cross-Validation Scores:
# Random Forest: The cross-validation scores for Random Forest show consistent high performance, indicating that the model is not overfitting and generalizes well to unseen data.
# Decision Tree, SVM, and KNN: These models have lower cross-validation scores compared to Random Forest, and they show more variability, suggesting less stability in their performance.
# Conclusion:
# The best model among the ones evaluated is the Random Forest.
# 
# Reasons:
# Highest Metrics: It achieves the highest accuracy, precision, recall, and F1 score among all models.
# Consistency: Its cross-validation scores are consistently high, indicating good generalization to new data.
# Balance: It provides a good balance between precision and recall, which is critical for classification tasks.
# The Random Forest model outperforms the other models in all key metrics, making it the best choice for this classification problem.

# ## HyperTuning

# In[32]:


param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'bootstrap': [True, False]
}

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                   n_iter=100, cv=3, n_jobs=-1, verbose=2, random_state=42)

# Fit the model using RandomizedSearchCV
random_search.fit(X_train_scaled, Y_train)

# Print the best parameters and estimator
print("Best parameters found: ", random_search.best_params_)
best_model = random_search.best_estimator_

# Make predictions on the test set using the best model
Y_test_prediction = best_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(Y_test, Y_test_prediction)
precision = precision_score(Y_test, Y_test_prediction, average='weighted')
recall = recall_score(Y_test, Y_test_prediction, average='weighted')
f1 = f1_score(Y_test, Y_test_prediction, average='weighted')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Perform cross-validation on the training set
cv_scores = cross_val_score(best_model, X_train_scaled, Y_train, cv=5)

print(f'Cross-validation scores: {cv_scores}')
print(f'Average cross-validation score: {cv_scores.mean():.2f}')


# ## Building a Predictive System

# In[37]:


import numpy as np

input_data = (7.5, 0.5, 0.36, 6.1, 0.071, 17.0, 102.0, 0.9978, 3.35, 0.8, 10.5)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = best_model.predict(input_data_reshaped)
print("Predicted Quality Score:", prediction[0])

if prediction[0] >= 6:
    print('Good Quality Wine')
else:
    print('Bad Quality Wine')

