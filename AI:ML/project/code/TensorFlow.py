
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pds
import matplotlib.pyplot as plt
from matplotlib import pylab
import seaborn as sns
import tensorflow as tf
sns.set_style("whitegrid")
# Reading data from .csv file.
noShow = pds.read_csv('/Users/surajsatheeshnair/Documents/NEU/AI/Project/Data/data_1.csv')
print(noShow.head())


# In[2]:

#Cleaning data for Age. 
#noShow = noShow[(noShow.Age > 0) & (noShow.Age < 100)]
maxAge , minAge = noShow.Age.max(), noShow.Age.min()
print (maxAge)
print (minAge)
# Normalizing column Age. describe gives us the statistics 
noShow.Age = noShow.Age.apply(lambda x: float(x-minAge)/(maxAge-minAge))
noShow.Age.describe()


# In[3]:

# Converting the column gender into numbers. Male - 0 and Female - 1
noShow.Gender = noShow.Gender.apply(lambda x: 0 if x =='M' else 1)
print (noShow.head())


# In[4]:

# Normalizing Awaiting Time from the dataset.
noShow.AwaitingTime = noShow.AwaitingTime.apply(abs)
minAwaitingTime = noShow.AwaitingTime.min()
maxAwaitingTime = noShow.AwaitingTime.max()
print (minAwaitingTime)
print (maxAwaitingTime)
noShow.AwaitingTime = noShow.AwaitingTime.apply(lambda x: float(x-minAwaitingTime)/(maxAwaitingTime-minAwaitingTime))
noShow.AwaitingTime.describe()


# In[5]:

# Converting Show-up or No-Showup status to 1 and 0 respectively
noShow.Status = noShow.Status.apply(lambda x: 1 if x =='Show-Up' else 0)


# In[6]:

# Days of the week should be in numbers and then normalizing the same.
list_wod = noShow.DayOfTheWeek.unique()
#print(list_wod)

dic_wod = {}
for i, e in enumerate(list_wod):
    dic_wod[e] = i

noShow.DayOfTheWeek = noShow.DayOfTheWeek.apply(lambda x: float(dic_wod[x])/len(list_wod))
# Here we can see that every field is in numbers now.
print(noShow.head())


# In[7]:

import re
pattern = re.compile(r'201(\d*)-(\d*)-(\d*).(\d*).*$')
arr_reg = np.ndarray([len(noShow), 4])
# Splitting Appointment Registration and Appintment Data fields into years, months, days and hours.
for i, e in enumerate(noShow.AppointmentRegistration):
    arr_reg[i] = list(re.search(pattern, e).groups())
df_reg = pds.DataFrame(arr_reg, columns=['reg_y', 'reg_m', 'reg_d', 'reg_h'])
# Normalizing the above
df_reg = (df_reg-df_reg.min())/(df_reg.max()-df_reg.min())

arr_apo = np.ndarray([len(noShow), 4])
for i, e in enumerate(noShow.ApointmentData):
    arr_apo[i] = list(re.search(pattern, e).groups())
df_apo = pds.DataFrame(arr_apo, columns=['apo_y', 'apo_m', 'apo_d', 'apo_h'])

df_apo = (df_apo-df_apo.min())/(df_apo.max()-df_apo.min())

noShow = noShow.join(df_reg)
noShow = noShow.join(df_apo)

noShow = noShow.drop(['AppointmentRegistration', 'ApointmentData'], axis=1)
noShow.head()


# In[8]:

labels = pds.get_dummies(noShow.Status, prefix='label')
noShow = noShow.join(labels)


# In[9]:

np.unique(arr_apo[:,-1])
noShow = noShow.drop('apo_h', axis=1)
noShow.info()
noShow.head()


# In[10]:

arr_data = noShow.as_matrix()
arr_data[:3]


# In[11]:

np.random.shuffle(arr_data)
#Splitting the data into equal chunks. 
# 300000 is the data entry size and splitting that into 5 chunks of 60000 each.
folds = np.split(arr_data, 6)
#Verifying the above.
len(folds), folds[0].shape


# In[12]:

# Tensor flow package.
g = tf.Graph()
with g.as_default():
    # Selecting all the columns as features. 
    # placeholder: A Tensor that may be used as a handle for feeding a value, but not evaluated directly.
    features = tf.placeholder(dtype=tf.float32, shape=[None, 20])
    labels = tf.placeholder(dtype=tf.float32, shape=[None, 2])
    w1 = tf.Variable(tf.truncated_normal(shape=[20, 10]))
    b1 = tf.Variable(tf.zeros(shape=[10]))
    
    n1 = tf.matmul(features, w1) + b1
    o1 = tf.nn.relu(n1)
    
    w2 = tf.Variable(tf.truncated_normal(shape=[10, 2]))
    b2 = tf.Variable(tf.zeros(shape=[2]))
    n2 = tf.matmul(o1, w2)+b2
    o2 = tf.nn.softmax(n2)
	#tf.train.Optimizer
	#tf.train.GradientDescentOptimizer
	#tf.train.AdadeltaOptimizer
	#tf.train.AdagradOptimizer
	#tf.train.AdagradDAOptimizer
	#tf.train.MomentumOptimizer
	#tf.train.AdamOptimizer
	#tf.train.FtrlOptimizer
	#tf.train.ProximalGradientDescentOptimizer
	#tf.train.ProximalAdagradOptimizer
	#tf.train.RMSPropOptimizer

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=n2)
    op_train = tf.train.GradientDescentOptimizer(0.000001).minimize(cross_entropy)
    op_init = tf.global_variables_initializer()
    
    correct_prediction = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, axis=1), tf.argmax(o2, axis=1)), tf.float32))
    print(correct_prediction)


# In[14]:

accrs = []

for i in range(6):
    print('start iteration #', i)
    
    train_features = np.concatenate([fold[:,:-2] for idx_fold, fold in enumerate(folds) if idx_fold!=i])
    train_labels = np.concatenate([fold[:, -2:] for idx_fold, fold in enumerate(folds) if idx_fold!=i])
    test_features, test_labels = folds[i][:,:-2], folds[i][:,-2:]
    
    with tf.Session(graph=g) as sess:
        sess.run(op_init)
        
        for j in range(500):
            _, cur_accr = sess.run([op_train, correct_prediction], feed_dict={features:train_features, labels:train_labels})
            
            if j%100 == 0 and j > 0:
                print('accuracy of #%d episodes: %f' % (j, np.mean(cur_accr)))
        
        accr = sess.run(correct_prediction, feed_dict={features:test_features, labels:test_labels})
        accrs.append(accr)
        print('accuracy of #%d iteration: %f' % (i, accr))
print('avg accuracy:', sum(accrs)/len(accrs))


# In[ ]:



