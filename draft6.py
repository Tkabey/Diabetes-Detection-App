#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pip
pip.main(["install","streamlit"])


# In[29]:


import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


st.header("Diabetes Detection App")


# In[6]:


image=Image.open("C:\\Users\\tkabe\\myprojects\\Diabetes\\diab.png")


# In[7]:


st.image(image)


# In[8]:


data=pd.read_csv("C:\\Users\\tkabe\\myprojects\\Diabetes\\diabetes.csv")
data.head()


# In[9]:


st.subheader("Data")


# In[10]:


st.dataframe(data)


# In[11]:


st.subheader("Data Description")


# In[12]:


st.write(data.iloc[:,:8].describe())


# In[ ]:





# In[ ]:





# In[13]:


data.isnull().sum()


# In[15]:


data.duplicated().sum()


# In[17]:


data.head(2)


# In[22]:


data["Outcome"].value_counts()


# In[30]:


sns.countplot(data["Outcome"])
plt.show()


# In[31]:


x=data.iloc[:,:8]
y=data.iloc[:,8]


# In[32]:


from sklearn.model_selection import train_test_split


# In[33]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[34]:


from sklearn.ensemble import RandomForestClassifier


# In[35]:


model=RandomForestClassifier(n_estimators=500)
model.fit(x_train,y_train)


# In[36]:


from sklearn.metrics import accuracy_score


# In[37]:


y_pred=model.predict(x_test)


# In[39]:


st.subheader("Accuracy of trained model")


# In[40]:


st.write(accuracy_score(y_test,y_pred))


# In[41]:


st.subheader("Enter your details")


# In[42]:


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


# In[43]:


ui=user_inputs()


# In[44]:


st.subheader("Entered Input Data")


# In[45]:


st.write(ui)


# In[46]:


data.corr()


# In[47]:


st.subheader("Predictions (0 - Non Diabetes, 1 - Diabetes)")


# In[48]:


st.write(model.predict(ui))


# In[49]:


if model.predict(ui)==1:
    st.info("You have Diabetes")
elif model.predict(ui)==0:
    st.info("You have no Diabetes")


# In[ ]:




