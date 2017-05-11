
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pds
import matplotlib.pyplot as plt
from matplotlib import pylab
import seaborn as sns
sns.set_style("whitegrid")
# Reading data from .csv file.
noShow = pds.read_csv('/Users/surajsatheeshnair/Documents/NEU/AI/Project/Data/data_1.csv')
print(noShow.head())


# In[3]:

# Data Cleaning
# Renaming the incorrectly named columns and keeping them in place. 
noShow.rename(columns = {'ApointmentData':'AppointmentData',
                         'Alcoolism': 'Alchoholism',
                         'HiperTension': 'Hypertension',
                         'Handcap': 'Handicap'}, inplace = True)

print(noShow.columns)


# In[4]:

noShow.AppointmentRegistration = noShow.AppointmentRegistration.apply(np.datetime64)
noShow.AppointmentData = noShow.AppointmentData.apply(np.datetime64)
noShow.AwaitingTime = noShow.AwaitingTime.apply(abs)


# In[5]:

def dayToNumber(day):
    if day == 'Monday': 
        return 0
    if day == 'Tuesday': 
        return 1
    if day == 'Wednesday': 
        return 2
    if day == 'Thursday': 
        return 3
    if day == 'Friday': 
        return 4
    if day == 'Saturday': 
        return 5
    if day == 'Sunday': 
        return 6

noShow.Gender = noShow.Gender.apply(lambda x: 1 if x == 'M' else 0)
noShow.DayOfTheWeek = noShow.DayOfTheWeek.apply(dayToNumber)
noShow.Status = noShow.Status.apply(lambda x: 1 if x == 'Show-Up' else 0)


features_train = noShow[['Age', 'Diabetes','Hypertension', 'Tuberculosis', 'Smokes',
                         'Alchoholism', 'Scholarship']].iloc[:297500]

labels_train = noShow.Status[:297500]

features_test = noShow[['Age', 'Diabetes','Hypertension', 'Tuberculosis', 'Smokes',
                         'Alchoholism', 'Scholarship']].iloc[297500:]

labels_test = noShow.Status[297500:]


from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

# MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

# clf =  MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(15,), learning_rate='constant',
       learning_rate_init=0.1, max_iter=1000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
clf = clf.fit(features_train, labels_train)
print('Accuracy:', round(accuracy_score(labels_test, 
                                        clf.predict(features_test)),1) * 100, '%')


# In[ ]:



