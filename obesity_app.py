import pandas as pd 
import numpy as np 
import joblib
import streamlit as st

df=pd.read_csv(ObesityDataSet_raw_and_data_sinthetic.csv')

df.head()

df.columns
# column modifcation
df.columns=['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
       'Feq_calory_food', 'Feq_vegatable_consumption', 'No_main_meals', 'snacks', 
       'Smoke', 'Daily_water_intake', 'Calory_monitoring', 'Physical_exercise', 'TUE',
       'Acohol_consumption', 'Mode_of_transit', 'Obesity_status']

df.isnull().sum() # no null values

df.duplicated().sum()
dfr=df.drop_duplicates(inplace=True) # dropping dupliacates

from sklearn .preprocessing import LabelEncoder
le=LabelEncoder()
df['Gender']=le.fit_transform(df['Gender'])
df['family_history_with_overweight']=le.fit_transform(df['family_history_with_overweight'])
df['Calory_monitoring']=le.fit_transform(df['Calory_monitoring'])
df['Smoke']=le.fit_transform(df['Smoke'])
df['Acohol_consumption']=le.fit_transform(df['Acohol_consumption'])
df['Mode_of_transit']=le.fit_transform(df['Mode_of_transit'])
df['Feq_calory_food']=le.fit_transform(df['Feq_calory_food'])
df['Obesity_status']=le.fit_transform(df['Obesity_status'])
df['snacks']=le.fit_transform(df['snacks'])




x=df[['Gender', 'Age','Weight', 'family_history_with_overweight',
       'Feq_calory_food', 'Feq_vegatable_consumption', 'No_main_meals',   
       'snacks', 'Smoke', 'Daily_water_intake', 'Calory_monitoring',      
       'Physical_exercise', 'Acohol_consumption']]
y=df['Obesity_status']

from sklearn.preprocessing import StandardScaler
from sklearn .model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn .linear_model import LogisticRegression
from sklearn .tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

scaler=StandardScaler()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

x_train.shape,x_test.shape,y_train.shape,y_test.shape

x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

tree=DecisionTreeClassifier()
knn=KNeighborsClassifier(3)
rn=RandomForestClassifier(n_estimators=200)
lg=LogisticRegression(max_iter=5)

tree.fit(x_train,y_train)
knn.fit(x_train,y_train)
rn.fit(x_train,y_train)
lg.fit(x_train,y_train)

tree.score(x_train,y_train)
knn.score(x_train,y_train)
rn.score(x_train,y_train)
lg.score(x_train,y_train)

#evaluations

tree_p=tree.predict(x_test)
knn_p=knn.predict(x_test)
rn_p=rn.predict(x_test)
lg_p=lg.predict(x_test)

from sklearn import metrics

metrics.accuracy_score(y_test,tree_p)

metrics.accuracy_score(y_test,knn_p)

metrics.accuracy_score(y_test,rn_p)

metrics.accuracy_score(y_test,lg_p)


# app building

joblib.dump(rn,'obesity_model.joblib') # model saving

def main():
    st.title('obesity/weight prediction')
    st.info('notifications: All inputs require numaricals values')

    Gender=st.text_input('Your gender (female=0,male=1)')
    Age= st.text_input('Age') 
    Weight=st.text_input('Input your weight')
    family_history_with_overweight=st.text_input('Do you have family history with overweight (yes=1,no=0)')
    Feq_calory_food=st.text_input('Do you eat calory foods (yes=1,no=0)')
    Feq_vegatable_consumption=st.text_input('How often do you eat vegetables (e.g,1,2,3,4,2.5)')
    No_main_meals=st.text_input('How many times do you eat daily?( e.g, 0,1,2,3,2.5,4)')  
    snacks=st.text_input('How often do you take snacks after meals? (e.g,1,0,2,3,2.8)') 
    Smoke=st.text_input('Do you smoke? (yes=1,no=0)')
    Daily_water_intake=st.text_input('How frequent do you take water daily? (e.g, 0,1,2,3,4)')
    Calory_monitoring=st.text_input('Do you monitor your calory intake? (yes=1,no=0)')      
    Physical_exercise=st.text_input('How often do you engage in physical activities? (e.g,0,1,2,3,4,5)')
    Acohol_consumption=st.text_input('How often do you take acohol daily? (e.g,0,1,2,3,4)')

    info=''

    input_data=np.array([[Gender,Age,Weight,family_history_with_overweight,
                         Feq_calory_food,Feq_vegatable_consumption,No_main_meals,   
                         snacks, Smoke,Daily_water_intake,Calory_monitoring,      
                         Physical_exercise, Acohol_consumption]])

    if st.button('weight category'):
         result=rn.predict(scaler.transform(input_data))

         if result==0:
            info=('insufficient weight')
         elif result==1:
            info=('normal weight')
         elif result==2:
            info=('Obesity_Type_I')
         elif result==3:
            info=('Obesity_Type_II')
         elif result==4:
            info=('Obesity_Type_III')
         elif result==5:
            info=('Overweight_Level_I')
         elif result==6:
            info=('Overweight_Level_II')
         else:
            info=('unknown')

         st.success(info)
   

if __name__=="__main__":
    main()

