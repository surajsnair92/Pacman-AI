import pandas as pd
from numpy import nan
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn import cross_validation

from sklearn import svm
from sklearn import decomposition
from sklearn.neighbors import KNeighborsClassifier

def run():
    train_data = pd.read_csv('./Data/ML Data Collection.csv')
    
    scalar = preprocessing.MinMaxScaler(feature_range=(0,1))
    
    #Mapping priority list into numbers
    priority_list = sorted(train_data['Priority'].unique())
    
    float_list = [float(i) for i in range(0,len(priority_list))]
    priority_mapping = dict(zip(priority_list,float_list))
    train_data['priority_val'] = train_data['Priority'].map(priority_mapping)
    
    train_data['priority_val'] = (scalar.fit_transform(train_data['priority_val'].reshape(-1, 1))).ravel([1])
    
    # APT normalization
    train_data['APT'] = (scalar.fit_transform(train_data['APT'].reshape(-1, 1))).ravel([1])
    
    # TSLR normalization
    train_data['TSLR'] = (scalar.fit_transform(train_data['TSLR'].reshape(-1, 1))).ravel([1])
    
    # Check if 'Calls from Customer' needs to be normalized
    floatCallsArray =  np.asarray([float(i) for i in train_data['Calls from Customer']])
    train_data['Calls from Customer'] = (scalar.fit_transform(floatCallsArray.reshape(-1, 1))).ravel([1])
    
    # Check if 'Number of Info to SAP from last TSLR' needs to be normalized
    floatNumInfoArray =  np.asarray([float(i) for i in train_data['Number of Info to SAP from last TSLR']])
    train_data['Number of Info to SAP from last TSLR'] = (scalar.fit_transform(floatNumInfoArray.reshape(-1, 1))).ravel([1])
    
    # Check if 'Frequent changes in processor' needs to be normalized - might not be, need to check the whole data
    floatFreqChangeArray =  np.asarray([float(i) for i in train_data['Frequent changes in processor']])
    train_data['Frequent changes in processor'] = (scalar.fit_transform(floatFreqChangeArray.reshape(-1, 1))).ravel([1])
    
    # Check if 'Number of Processors' needs to be normalized - might not be, need to check the whole data
    floatNumProcArray =  np.asarray([float(i) for i in train_data['Number of Processors']])
    train_data['Number of Processors'] = (scalar.fit_transform(floatNumProcArray.reshape(-1, 1))).ravel([1])
    
    #Mapping core business list into numbers
    core_business_list = sorted(train_data['Core Business affected?'].unique())
    core_business_list.remove(nan)
    core_business_mapping = dict(zip(core_business_list,range(0,len(core_business_list))))
    train_data['core_business_val'] = train_data['Core Business affected?'].map(core_business_mapping)
    
    train_data['core_business_val'] = train_data['core_business_val'].fillna(train_data['core_business_val'].median())
    
    
    #Mapping Go live list into numbers
    go_live_list = sorted(train_data['Go live Affected?'].unique())
    go_live_list.remove(nan)
    go_live_mapping = dict(zip(go_live_list,range(0,len(go_live_list))))
    train_data['go_live_val'] = train_data['Go live Affected?'].map(go_live_mapping)
    
    train_data['go_live_val'] = train_data['go_live_val'].fillna(train_data['go_live_val'].median())
    
    #Mapping Work around list into numbers
    work_around_list = sorted(train_data['Work around available?'].unique())
    work_around_list.remove(nan)
    work_around_mapping = dict(zip(work_around_list,range(0,len(work_around_list))))
    train_data['work_around_val'] = train_data['Work around available?'].map(work_around_mapping)
    
    train_data['work_around_val'] = train_data['work_around_val'].fillna(train_data['work_around_val'].median())
    
    #Mapping escalation states into numbers
    escStages = sorted(train_data['Stage of Escalation'].unique())
    states_mapping = dict(zip(escStages,range(0,len(escStages))))
    train_data['escStages_val'] = train_data['Stage of Escalation'].map(states_mapping)
    
    train_data = train_data.drop(['Stage of Escalation', 'Priority', 'Customer call back request', 'Core Business affected?',
                                 'Go live Affected?', 'Work around available?'], axis=1)
                                 
    train_data = train_data.drop(['Num of users affected', 'Business Impact mentioned?'], axis=1)
    
    #removing to test
    train_data = train_data.drop(['core_business_val','work_around_val'], axis=1)
    
    train_values = train_data.values
    X = train_values[:,:-1]
    Y = train_values[:,-1:]
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,Y, test_size=0.33, random_state=0)
    
    rf = RandomForestClassifier(n_estimators=870, min_samples_leaf =8, n_jobs = -1, max_depth=12)
    rf.fit(X_train,y_train.ravel([1]))
    
    
    #=======================================================
    #for Demo ******************************************
    #=======================================================
    
    train_data_d = pd.read_csv('./Data/Demo_data.csv')
    
    scalar_d = preprocessing.MinMaxScaler(feature_range=(0,1))
    
    #Mapping priority list into numbers
    priority_list_d = sorted(train_data_d['Priority'].unique())
    
    float_list_d = [float(i) for i in range(0,len(priority_list_d))]
    priority_mapping_d = dict(zip(priority_list_d,float_list_d))
    train_data_d['priority_val'] = train_data_d['Priority'].map(priority_mapping_d)
    
    train_data_d['priority_val'] = (scalar_d.fit_transform(train_data_d['priority_val'].reshape(-1, 1))).ravel([1])
    
    # APT normalization
    train_data_d['APT'] = (scalar_d.fit_transform(train_data_d['APT'].reshape(-1, 1))).ravel([1])
    
    # TSLR normalization
    train_data_d['TSLR'] = (scalar_d.fit_transform(train_data_d['TSLR'].reshape(-1, 1))).ravel([1])
    
    # Check if 'Calls from Customer' needs to be normalized
    floatCallsArray_d =  np.asarray([float(i) for i in train_data_d['Calls from Customer']])
    train_data_d['Calls from Customer'] = (scalar_d.fit_transform(floatCallsArray_d.reshape(-1, 1))).ravel([1])
    
    # Check if 'Number of Info to SAP from last TSLR' needs to be normalized
    floatNumInfoArray_d =  np.asarray([float(i) for i in train_data_d['Number of Info to SAP from last TSLR']])
    train_data_d['Number of Info to SAP from last TSLR'] = (scalar_d.fit_transform(floatNumInfoArray_d.reshape(-1, 1))).ravel([1])
    
    # Check if 'Frequent changes in processor' needs to be normalized - might not be, need to check the whole data
    floatFreqChangeArray_d =  np.asarray([float(i) for i in train_data_d['Frequent changes in processor']])
    train_data_d['Frequent changes in processor'] = (scalar_d.fit_transform(floatFreqChangeArray_d.reshape(-1, 1))).ravel([1])
    
    # Check if 'Number of Processors' needs to be normalized - might not be, need to check the whole data
    floatNumProcArray_d =  np.asarray([float(i) for i in train_data_d['Number of Processors']])
    train_data_d['Number of Processors'] = (scalar_d.fit_transform(floatNumProcArray_d.reshape(-1, 1))).ravel([1])
    
    #Mapping core business list into numbers
    core_business_list_d = sorted(train_data_d['Core Business affected?'].unique())
    core_business_list_d.remove(nan)
    core_business_mapping_d = dict(zip(core_business_list_d,range(0,len(core_business_list_d))))
    train_data_d['core_business_val'] = train_data_d['Core Business affected?'].map(core_business_mapping_d)
    
    train_data_d['core_business_val'] = train_data_d['core_business_val'].fillna(train_data_d['core_business_val'].median())
    
    
    #Mapping Go live list into numbers
    go_live_list_d = sorted(train_data_d['Go live Affected?'].unique())
    go_live_list_d.remove(nan)
    go_live_mapping_d = dict(zip(go_live_list_d,range(0,len(go_live_list_d))))
    train_data_d['go_live_val'] = train_data_d['Go live Affected?'].map(go_live_mapping_d)
    
    train_data_d['go_live_val'] = train_data_d['go_live_val'].fillna(train_data_d['go_live_val'].median())
    
    #Mapping Work around list into numbers
    work_around_list_d = sorted(train_data_d['Work around available?'].unique())
    work_around_list_d.remove(nan)
    work_around_mapping_d = dict(zip(work_around_list_d,range(0,len(work_around_list_d))))
    train_data_d['work_around_val'] = train_data_d['Work around available?'].map(work_around_mapping_d)
    
    train_data_d['work_around_val'] = train_data_d['work_around_val'].fillna(train_data_d['work_around_val'].median())
    
    #Mapping escalation states into numbers
    escStages_d = sorted(train_data_d['Stage of Escalation'].unique())
    states_mapping_d = dict(zip(escStages_d,range(0,len(escStages_d))))
    train_data_d['escStages_val'] = train_data_d['Stage of Escalation'].map(states_mapping_d)
    
    train_data_d = train_data_d.drop(['Stage of Escalation', 'Priority', 'Customer call back request', 'Core Business affected?',
                                 'Go live Affected?', 'Work around available?'], axis=1)
                                 
    train_data_d = train_data_d.drop(['Num of users affected', 'Business Impact mentioned?'], axis=1)
    
    #removing to test
    train_data_d = train_data_d.drop(['core_business_val','work_around_val'], axis=1)     
    
    demo_values = train_data_d.values
    
    test_row_values = demo_values[-7:,:]
    test_values = test_row_values[:,:-1]
    
    #print to the file
    fileName = 'algoResult.txt'
    sliderFile = 'sliderInput.txt'
    
    algofile =  open(fileName, 'w')
    algofile.truncate()
    
    sliderfile =  open(sliderFile, 'w')
    sliderfile.truncate()
    
    predicted_Y = rf.predict(test_values)
    
    for p_value in predicted_Y:
        algofile.write(repr(int(p_value))+'\n')
    for slider in test_values:
        for s_value in slider:
            sliderfile.write(repr(round(float(s_value),2))+',')
        sliderfile.write('\n')
        
    algofile.close()  
    sliderfile.close()
    

def main():
    run()
 
if __name__ == '__main__':
    main()                         