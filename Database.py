
# coding: utf-8

# <br>
# ### 1) Importing necessary libraries

# In[16]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True) #to keep the values in decimals


# <br>
# ### 2) Loading the dataset and dividing it into training and testing set

# In[5]:


def database():
    data = pd.read_csv('dataset.csv')  
    data = data.sample(frac=1).reset_index(drop=True)
    training_set = data[:10000]
    test_set = data[10000:12330].reset_index(drop=True)
    return training_set,test_set


# <br>
# ### 3) Converting non- numeric attributes to numeric attributes

# In[6]:


# for special day
def calc_spl(data):
    n0 = (data.loc[(data['SpecialDay']==0.0) & (data['Revenue']==True)].shape[0])/(data.loc[data['SpecialDay']==0.0].shape[0])
    n2 = (data.loc[(data['SpecialDay']==0.2) & (data['Revenue']==True)].shape[0])/(data.loc[data['SpecialDay']==0.2].shape[0])
    n4 = (data.loc[(data['SpecialDay']==0.4) & (data['Revenue']==True)].shape[0])/(data.loc[data['SpecialDay']==0.4].shape[0])
    n6 = (data.loc[(data['SpecialDay']==0.6) & (data['Revenue']==True)].shape[0])/(data.loc[data['SpecialDay']==0.6].shape[0])
    n8 = (data.loc[(data['SpecialDay']==0.8) & (data['Revenue']==True)].shape[0])/(data.loc[data['SpecialDay']==0.8].shape[0])
    n1 = (data.loc[(data['SpecialDay']==1.0) & (data['Revenue']==True)].shape[0])/(data.loc[data['SpecialDay']==1.0].shape[0])
    return n0,n2,n4,n6,n8,n1
    

# for month
def calc_month(data):
    feb = (data.loc[(data['Month']=='Feb') & (data['Revenue']==True)].shape[0])/(data.loc[data['Month']=='Feb'].shape[0])
    mar = (data.loc[(data['Month']=='Mar') & (data['Revenue']==True)].shape[0])/(data.loc[data['Month']=='Mar'].shape[0])
    may = (data.loc[(data['Month']=='May') & (data['Revenue']==True)].shape[0])/(data.loc[data['Month']=='May'].shape[0])
    jun = (data.loc[(data['Month']=='June') & (data['Revenue']==True)].shape[0])/(data.loc[data['Month']=='June'].shape[0])
    jul = (data.loc[(data['Month']=='Jul') & (data['Revenue']==True)].shape[0])/(data.loc[data['Month']=='Jul'].shape[0])
    aug = (data.loc[(data['Month']=='Aug') & (data['Revenue']==True)].shape[0])/(data.loc[data['Month']=='Aug'].shape[0])
    sep = (data.loc[(data['Month']=='Sep') & (data['Revenue']==True)].shape[0])/(data.loc[data['Month']=='Sep'].shape[0])
    octo = (data.loc[(data['Month']=='Oct') & (data['Revenue']==True)].shape[0])/(data.loc[data['Month']=='Oct'].shape[0])
    nov = (data.loc[(data['Month']=='Nov') & (data['Revenue']==True)].shape[0])/(data.loc[data['Month']=='Nov'].shape[0])
    dec = (data.loc[(data['Month']=='Dec') & (data['Revenue']==True)].shape[0])/(data.loc[data['Month']=='Dec'].shape[0])
    return feb,mar,may,jun,jul,aug,sep,octo,nov,dec


# for operating system
def calc_os(data):
    os1 = (data.loc[(data['OperatingSystems']==1) & (data['Revenue']==True)].shape[0])/(data.loc[data['OperatingSystems']==1].shape[0])
    os2 = (data.loc[(data['OperatingSystems']==2) & (data['Revenue']==True)].shape[0])/(data.loc[data['OperatingSystems']==2].shape[0])
    os3 = (data.loc[(data['OperatingSystems']==3) & (data['Revenue']==True)].shape[0])/(data.loc[data['OperatingSystems']==3].shape[0])
    os4 = (data.loc[(data['OperatingSystems']==4) & (data['Revenue']==True)].shape[0])/(data.loc[data['OperatingSystems']==4].shape[0])
    os5 = (data.loc[(data['OperatingSystems']==5) & (data['Revenue']==True)].shape[0])/(data.loc[data['OperatingSystems']==5].shape[0])
    os6 = (data.loc[(data['OperatingSystems']==6) & (data['Revenue']==True)].shape[0])/(data.loc[data['OperatingSystems']==6].shape[0])
    os7 = (data.loc[(data['OperatingSystems']==7) & (data['Revenue']==True)].shape[0])/(data.loc[data['OperatingSystems']==7].shape[0])
    os8 = (data.loc[(data['OperatingSystems']==8) & (data['Revenue']==True)].shape[0])/(data.loc[data['OperatingSystems']==8].shape[0])
    return os1,os2,os3,os4,os5,os6,os7,os8
    

# for browser
def calc_browser(data):
    b1 = (data.loc[(data['Browser']==1) & (data['Revenue']==True)].shape[0])/(data.loc[data['Browser']==1].shape[0])
    b2 = (data.loc[(data['Browser']==2) & (data['Revenue']==True)].shape[0])/(data.loc[data['Browser']==2].shape[0])
    b3 = (data.loc[(data['Browser']==3) & (data['Revenue']==True)].shape[0])/(data.loc[data['Browser']==3].shape[0])
    b4 = (data.loc[(data['Browser']==4) & (data['Revenue']==True)].shape[0])/(data.loc[data['Browser']==4].shape[0])
    b5 = (data.loc[(data['Browser']==5) & (data['Revenue']==True)].shape[0])/(data.loc[data['Browser']==5].shape[0])
    b6 = (data.loc[(data['Browser']==6) & (data['Revenue']==True)].shape[0])/(data.loc[data['Browser']==6].shape[0])
    b7 = (data.loc[(data['Browser']==7) & (data['Revenue']==True)].shape[0])/(data.loc[data['Browser']==7].shape[0])
    b8 = (data.loc[(data['Browser']==8) & (data['Revenue']==True)].shape[0])/(data.loc[data['Browser']==8].shape[0])
    b9 = (data.loc[(data['Browser']==9) & (data['Revenue']==True)].shape[0])/(data.loc[data['Browser']==9].shape[0])
    b10 = (data.loc[(data['Browser']==10) & (data['Revenue']==True)].shape[0])/(data.loc[data['Browser']==10].shape[0])
    b11 = (data.loc[(data['Browser']==11) & (data['Revenue']==True)].shape[0])/(data.loc[data['Browser']==11].shape[0])
    return b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11


# for region
def calc_region(data):
    r1 = (data.loc[(data['Region']==1) & (data['Revenue']==True)].shape[0])/(data.loc[data['Region']==1].shape[0])
    r2 = (data.loc[(data['Region']==2) & (data['Revenue']==True)].shape[0])/(data.loc[data['Region']==2].shape[0])
    r3 = (data.loc[(data['Region']==3) & (data['Revenue']==True)].shape[0])/(data.loc[data['Region']==3].shape[0])
    r4 = (data.loc[(data['Region']==4) & (data['Revenue']==True)].shape[0])/(data.loc[data['Region']==4].shape[0])
    r5 = (data.loc[(data['Region']==5) & (data['Revenue']==True)].shape[0])/(data.loc[data['Region']==5].shape[0])
    r6 = (data.loc[(data['Region']==6) & (data['Revenue']==True)].shape[0])/(data.loc[data['Region']==6].shape[0])
    r7 = (data.loc[(data['Region']==7) & (data['Revenue']==True)].shape[0])/(data.loc[data['Region']==7].shape[0])
    r8 = (data.loc[(data['Region']==8) & (data['Revenue']==True)].shape[0])/(data.loc[data['Region']==8].shape[0])
    r9 = (data.loc[(data['Region']==9) & (data['Revenue']==True)].shape[0])/(data.loc[data['Region']==9].shape[0])
    return r1,r2,r3,r4,r5,r6,r7,r8,r9


# for traffic type 
def calc_trafftype(data):
    t1 = (data.loc[(data['TrafficType']==1) & (data['Revenue']==True)].shape[0])/(data.loc[data['TrafficType']==1].shape[0])
    t2 = (data.loc[(data['TrafficType']==2) & (data['Revenue']==True)].shape[0])/(data.loc[data['TrafficType']==2].shape[0])
    t3 = (data.loc[(data['TrafficType']==3) & (data['Revenue']==True)].shape[0])/(data.loc[data['TrafficType']==3].shape[0])
    t4 = (data.loc[(data['TrafficType']==4) & (data['Revenue']==True)].shape[0])/(data.loc[data['TrafficType']==4].shape[0])
    t5 = (data.loc[(data['TrafficType']==5) & (data['Revenue']==True)].shape[0])/(data.loc[data['TrafficType']==5].shape[0])
    t6 = (data.loc[(data['TrafficType']==6) & (data['Revenue']==True)].shape[0])/(data.loc[data['TrafficType']==6].shape[0])
    t7 = (data.loc[(data['TrafficType']==7) & (data['Revenue']==True)].shape[0])/(data.loc[data['TrafficType']==7].shape[0])
    t8 = (data.loc[(data['TrafficType']==8) & (data['Revenue']==True)].shape[0])/(data.loc[data['TrafficType']==8].shape[0])
    t9 = (data.loc[(data['TrafficType']==9) & (data['Revenue']==True)].shape[0])/(data.loc[data['TrafficType']==9].shape[0])
    t10 = (data.loc[(data['TrafficType']==10) & (data['Revenue']==True)].shape[0])/(data.loc[data['TrafficType']==10].shape[0])
    t11 = (data.loc[(data['TrafficType']==11) & (data['Revenue']==True)].shape[0])/(data.loc[data['TrafficType']==11].shape[0])
    t12 = (data.loc[(data['TrafficType']==12) & (data['Revenue']==True)].shape[0])/(data.loc[data['TrafficType']==12].shape[0])
    t13 = (data.loc[(data['TrafficType']==13) & (data['Revenue']==True)].shape[0])/(data.loc[data['TrafficType']==13].shape[0])
    t14 = (data.loc[(data['TrafficType']==14) & (data['Revenue']==True)].shape[0])/(data.loc[data['TrafficType']==14].shape[0])
    t15 = (data.loc[(data['TrafficType']==15) & (data['Revenue']==True)].shape[0])/(data.loc[data['TrafficType']==15].shape[0])
    t16 = (data.loc[(data['TrafficType']==16) & (data['Revenue']==True)].shape[0])/(data.loc[data['TrafficType']==16].shape[0])
    t17 = (data.loc[(data['TrafficType']==17) & (data['Revenue']==True)].shape[0])/(data.loc[data['TrafficType']==17].shape[0])
    t18 = (data.loc[(data['TrafficType']==18) & (data['Revenue']==True)].shape[0])/(data.loc[data['TrafficType']==18].shape[0])
    t19 = (data.loc[(data['TrafficType']==19) & (data['Revenue']==True)].shape[0])/(data.loc[data['TrafficType']==19].shape[0])
    t20 = (data.loc[(data['TrafficType']==20) & (data['Revenue']==True)].shape[0])/(data.loc[data['TrafficType']==20].shape[0])
    return t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20


# for visitor type
def calc_visitortype(data):
    v1 = (data.loc[(data['VisitorType']=='Returning_Visitor') & (data['Revenue']==True)].shape[0])/(data.loc[data['VisitorType']=='Returning_Visitor'].shape[0])
    v2 = (data.loc[(data['VisitorType']=='New_Visitor') & (data['Revenue']==True)].shape[0])/(data.loc[data['VisitorType']=='New_Visitor'].shape[0])
    v3 = (data.loc[(data['VisitorType']=='Other') & (data['Revenue']==True)].shape[0])/(data.loc[data['VisitorType']=='Other'].shape[0])
    return v1,v2,v3


# for weekdays/weekends
def calc_weekend(data):
    wt1 = (data.loc[(data['Weekend']==True) & (data['Revenue']==True)].shape[0])/(data.loc[data['Weekend']==True].shape[0])
    wt2 = (data.loc[(data['Weekend']==False) & (data['Revenue']==True)].shape[0])/(data.loc[data['Weekend']==False].shape[0])
    return wt1,wt2


# <br>
# ### 4) The main function

# In[7]:


def main():
    # loading the database
    training_set,test_set=database()
    
    # accumulating all the variables
    n0,n2,n4,n6,n8,n1 = calc_spl(training_set)
    
    feb,mar,may,jun,jul,aug,sep,octo,nov,dec = calc_month(training_set)
    
    os1,os2,os3,os4,os5,os6,os7,os8 = calc_os(training_set)
    
    b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11 = calc_browser(training_set)
    
    r1,r2,r3,r4,r5,r6,r7,r8,r9 = calc_region(training_set)
    
    t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20 = calc_trafftype(training_set)
    
    v1,v2,v3 = calc_visitortype(training_set)
    
    wt1,wt2 = calc_weekend(training_set)
    
    # replacing the values 
    training_set['SpecialDay']= training_set['SpecialDay'].replace([0.0,0.2,0.4,0.6,0.8,1.0],[n0,n2,n4,n6,n8,n1])
    test_set['SpecialDay']= test_set['SpecialDay'].replace([0.0,0.2,0.4,0.6,0.8,1.0],[n0,n2,n4,n6,n8,n1])
    
    training_set['Month']= training_set['Month'].replace(['Feb','Mar','May','June','Jul','Aug','Sep','Oct','Nov','Dec'],[feb,mar,may,jun,jul,aug,sep,octo,nov,dec])
    test_set['Month']= test_set['Month'].replace(['Feb','Mar','May','June','Jul','Aug','Sep','Oct','Nov','Dec'],[feb,mar,may,jun,jul,aug,sep,octo,nov,dec])
    
    training_set['OperatingSystems']= training_set['OperatingSystems'].replace([1,2,3,4,5,6,7,8],[os1,os2,os3,os4,os5,os6,os7,os8])
    test_set['OperatingSystems']= test_set['OperatingSystems'].replace([1,2,3,4,5,6,7,8],[os1,os2,os3,os4,os5,os6,os7,os8])
    
    training_set['Browser']= training_set['Browser'].replace([1,2,3,4,5,6,7,8,9,10,11],[b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11])
    test_set['Browser']= test_set['Browser'].replace([1,2,3,4,5,6,7,8,9,10,11],[b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11])
    
    training_set['Region']= training_set['Region'].replace([1,2,3,4,5,6,7,8,9],[r1,r2,r3,r4,r5,r6,r7,r8,r9])
    test_set['Region']= test_set['Region'].replace([1,2,3,4,5,6,7,8,9],[r1,r2,r3,r4,r5,r6,r7,r8,r9])
    
    training_set['TrafficType']= training_set['TrafficType'].replace([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],[t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20])
    test_set['TrafficType']= test_set['TrafficType'].replace([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],[t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20])
    
    training_set['VisitorType']= training_set['VisitorType'].replace(['Returning_Visitor','New_Visitor','Other'],[v1,v2,v3])
    test_set['VisitorType']= test_set['VisitorType'].replace(['Returning_Visitor','New_Visitor','Other'],[v1,v2,v3])
    
    training_set['Weekend']= training_set['Weekend'].replace([True,False],[wt1,wt2])
    test_set['Weekend']= test_set['Weekend'].replace([True,False],[wt1,wt2])
    
    training_set['Revenue']= training_set['Revenue'].replace([True,False],[1.0,-1.0])
    test_set['Revenue']= test_set['Revenue'].replace([True,False],[1.0,-1.0])
    
    # converting values to numpy
    set1 = training_set.to_numpy()
    
    #splitting values as inputs and outputs
    X_train = set1[:,:17]
    Y_train = set1[:,17]
        
    set1 = test_set.to_numpy()
    X_test = set1[:,:17]
    Y_test = set1[:,17]
    return X_train,Y_train,X_test,Y_test


# <br>
# ### Calling the main function

# In[10]:


if __name__=='__main__':
    main()


# <br><br>
# ## Data Visualisation

# ##### Number of users who purchased and did not purchase products

# In[39]:


bought = ['Yes', 'No']
no = [1908,10422]
plt.bar(bought,no,color=['y','c'])
plt.show()


# <br>
# ##### Number of users who purchased on a special day vs number of users who purchased on a regular day

# In[36]:


day = ['Special Day', 'Regular Day']
no = [77,1831]
plt.bar(day,no,color=['r','g'])
plt.show()


# <br>
# ##### Number of purchases ever month

# In[26]:


month = ['Feb','Mar','May','June','Jul','Aug','Sep','Oct','Nov','Dec']
no = [3,192,365,29,66,76,86,115,760,216]
plt.pie(no,labels = month,autopct='%1.2f%%')
plt.show()


# <br>
# ##### Number of users from each region

# In[28]:


region = [1,2,3,4,5,6,7,8,9]
no = [4780,1136,2403,1182,318,805,761,434,511]
plt.pie(no,labels = region,autopct='%1.2f%%')
plt.show()


# <br>
# ##### Number of users who visit online shopping sites on weekends and weekdays

# In[32]:


day = ['Weekend', 'Weekday']
no = [2868,9462]
plt.bar(day,no,color=['y','g'])
plt.show()

