
# coding: utf-8

# In[166]:

import numpy as np
import pandas as pds
import matplotlib.pyplot as plt
from matplotlib import pylab
import seaborn as sns
sns.set_style("whitegrid")
# Reading data from .csv file.
noShow = pds.read_csv('/Users/surajsatheeshnair/Documents/NEU/AI/Project/Data/data_1.csv')
print(noShow.head())


# In[167]:

# Data Cleaning
# Renaming the incorrectly named columns and keeping them in place. 
noShow.rename(columns = {'ApointmentData':'AppointmentData',
                         'Alcoolism': 'Alchoholism',
                         'HiperTension': 'Hypertension',
                         'Handcap': 'Handicap'}, inplace = True)

print(noShow.columns)


# In[168]:

# Time in Appointment Registration and Appointment Data should be uniform. 
# Converting both to datetime64 format. 
# .head() gives the first 5 rows from the .csv file which is now in noShow. 
noShow.AppointmentRegistration = noShow.AppointmentRegistration.apply(np.datetime64)
noShow.AppointmentData = noShow.AppointmentData.apply(np.datetime64)
noShow.AwaitingTime = noShow.AwaitingTime.apply(abs)

print(noShow.AppointmentRegistration.head())
print(noShow.AppointmentData.head())
print(noShow.AwaitingTime.head())


# In[169]:

# Awaiting time is the number of days after one registers an appointment to the appointment date.
# Appointment Registration, Appointment Data and Awaiting time could be the features helping us predict noShow. 
from datetime import datetime as dt
numberOfDaysToAppointment =  noShow.AppointmentData - noShow.AppointmentRegistration
numberOfDaysToAppointment = numberOfDaysToAppointment.apply(lambda x: x.total_seconds() / (3600 * 24))


plt.scatter(noShow.AwaitingTime, numberOfDaysToAppointment)
plt.axis([0, 400, 0, 400])
plt.xlabel('Awaiting Time')
plt.ylabel('Number of Days to appointment')
plt.show()


# In[170]:

# hourCalculation is a function which tells us at what time of the day is the appointment fixed. 
# We do this calculation from AppointmentRegistration column.
def hourCalculation(time):
    time = str(time)
    hour = int(time[11:13])
    minute = int(time[14:16])
    seconds = int(time[17:19])
    totalHours = hour + (minute/60) + (seconds/3600)
    return round(totalHours)
noShow['hourCalc'] = noShow.AppointmentRegistration.apply(hourCalculation)
print (noShow.hourCalc.head())


# In[171]:

# Data Cleaning
# Removing outliers.
# Check outliers for all the required fields.
# We see that there are negative ages which is to be removed as age cannot be negative.
# Also, ages above 100 can be removed as this might alter the prediction. 

#print (noShow.Age.tail())
noShow = noShow[(noShow.Age > 0) & (noShow.Age < 100)]

# Checking outliers for Awaiting Time as this could be an important feature for prediction. 
# X-axis has awaiting time where we can see that above 350 there is only one point. So removing this noise. 
noShow = noShow[noShow.AwaitingTime < 350]


# In[172]:

# Determining features for prediction. All noises are cleared.
def probStatus(dataset, group_by):
    df = pds.crosstab(index = dataset[group_by], columns = dataset.Status).reset_index()
    df['probShowUp'] = df['Show-Up'] / (df['Show-Up'] + df['No-Show'])
    return df[[group_by, 'probShowUp']]


# In[173]:

sns.lmplot(data = probStatus(noShow, 'Age'), x = 'Age', y = 'probShowUp', fit_reg = True)
sns.plt.xlim(0, 100) # We have set the age to be < 100
sns.plt.title('Probability of showing up with respect to Age')
sns.plt.show()


# In[174]:

sns.lmplot(data = probStatus(noShow, 'hourCalc'), x = 'hourCalc', 
           y = 'probShowUp', fit_reg = True)
sns.plt.title('Probability of showing up with respect to HourOfTheDay')
sns.plt.show()


# In[175]:

# From the plots, we can conclude that Age can be a feature for prediction but not the hour of the appointment.
def probStatusCategorical(group_by):
    rows = []
    for item in group_by:
        for level in noShow[item].unique():
            row = {'Condition': item}
            total = len(noShow[noShow[item] == level])
            n = len(noShow[(noShow[item] == level) & (noShow.Status == 'Show-Up')])
            row.update({'Level': level, 'Probability': n / total})
            rows.append(row)
    return pds.DataFrame(rows)

sns.barplot(data = probStatusCategorical(['Diabetes', 'Alchoholism', 'Hypertension',
                                         'Tuberculosis', 'Smokes', 'Scholarship']),
            x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2')
sns.plt.title('Probability of showing up')
sns.plt.ylabel('Probability')
sns.plt.show()


# In[176]:

# Analyzing if Gender plays any role in appointment show up or not.
sns.barplot(data = probStatusCategorical(['Gender']),
            x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2')
sns.plt.title('Probability of showing up')
sns.plt.ylabel('Probability')
sns.plt.show()


# In[177]:

sns.barplot(data = probStatusCategorical(['Handicap']),
            x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2')
sns.plt.title('Probability of showing up')
sns.plt.ylabel('Probability')
sns.plt.show()


# In[178]:

# Analyzing if days of the week affects showup or not.
sns.barplot(data = probStatusCategorical(['DayOfTheWeek']),
            x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2',
           hue_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                       'Saturday', 'Sunday'])
sns.plt.title('Probability of showing up')
sns.plt.ylabel('Probability')
sns.plt.show()


# In[179]:

#SMS Reminder as a feature.
sns.barplot(data = probStatusCategorical(['Sms_Reminder']),
            x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2')
sns.plt.title('Probability of showing up')
sns.plt.ylabel('Probability')
sns.plt.show()

