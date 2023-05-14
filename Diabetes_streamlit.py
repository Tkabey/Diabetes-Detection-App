#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pip
pip.main(["install","streamlit"])


# In[32]:


import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np


# In[3]:


st.header("Diabetes Detection App")


# In[4]:


image=Image.open("C:\\Users\\tkabe\\myprojects\\Diabetes\\Diabetes-Detection-App\\diab.png")


# In[5]:


st.image(image)


# In[6]:


data=pd.read_csv("C:\\Users\\tkabe\\myprojects\\Diabetes\\Diabetes-Detection-App\\diabetes.csv")
data.head()


# In[7]:


st.subheader("Data")


# In[8]:


st.dataframe(data)


# In[9]:


st.subheader("Data Description")


# In[10]:


st.write(data.iloc[:,:8].describe())


# In[11]:


data.isnull().sum()


# In[12]:


data.duplicated().sum()


# In[13]:


data["Outcome"].value_counts()


# In[15]:


data.corr()


# In[ ]:





# In[ ]:





# In[16]:


x=data.iloc[:,:8]
y=data.iloc[:,8]


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[19]:


from sklearn.ensemble import RandomForestClassifier


# In[20]:


model=RandomForestClassifier(n_estimators=500)
model.fit(x_train,y_train)


# In[21]:


from sklearn.metrics import accuracy_score


# In[22]:


y_pred=model.predict(x_test)


# In[23]:


st.subheader("Accuracy of trained model")


# In[24]:


st.write(accuracy_score(y_test,y_pred))


# In[25]:


st.subheader("Enter your details")


# In[26]:


def user_inputs():
    preg=st.slider("Pregnancies",0,20,0)
    glu=st.slider("Glucose",0,200,0)
    bp=st.slider("Blood Pressure",0,130,0)
    sthick=st.slider("Skin Thickness",0,100,0)
    ins=st.slider("Insulin",0.0,1000.0,0.0)
    bmi=st.slider("BMI",0.0,70.0,0.0)
    dpf=st.slider("DPF",0.000,3.000,0.000)
    age=st.slider("Age",0,100,0)
 
    input_dict={"Pregnancies":preg,
                "Glucose":glu,
                "Blood Pressure":bp,
                "Skin Thickness":sthick,
                "Insulin":ins,
                "BMI":bmi,
                "DPF":dpf,
                "Age":age}
    return pd.DataFrame(input_dict,index=[0])


# In[27]:


ui=user_inputs()


# In[28]:


st.subheader("Entered Input Data")


# In[29]:


st.write(ui)


# In[30]:


st.subheader("Predictions (0 - Non Diabetes, 1 - Diabetes)")


# In[31]:


st.write(model.predict(ui))


# In[49]:


if model.predict(ui)==1:
    st.info("You have Diabetes")
elif model.predict(ui)==0:
    st.info("You have no Diabetes")


# In[ ]:




