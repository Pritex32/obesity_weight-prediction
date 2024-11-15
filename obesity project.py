#!/usr/bin/env python
# coding: utf-8

# # Analysis by Prisca 

# # About the Dataset:
# ## This dataset include data for the estimation of obesity levels in individuals from the countries of Mexico, Peru and Colombia, based on their eating habits and physical condition. 
# ## The data contains 17 attributes and 2111 records, the records are labeled with the class variable such as NObesity (Obesity Level), that allows classification of the data using the values of Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III.

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_csv('C:\\Users\\USER\\Documents\\dataset\\ObesityDataSet_raw_and_data_sinthetic.csv')


# In[4]:


df.head()


# In[ ]:





# # Data Cleaning

# In[5]:


df.columns


# In[6]:


df.shape


# In[7]:


is_null=df.isnull().sum() # checking for missing values


# In[8]:


is_null


# In[9]:


is_dup=df.duplicated().sum() #checking for duplicates


# In[10]:


is_dup # 24 duplicates found


# # Handling duplicates

# In[11]:


dfr=df.drop_duplicates(inplace=True)


# In[12]:


df['CAEC'].unique()


# ## column renaming

# In[13]:


df.columns=['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
       'Feq_calory_food', 'Feq_vegatable_consumption', 'No_main_meals', 'snacks', 
       'Smoke', 'Daily_water_intake', 'Calory_monitoring', 'Physical_exercise', 'TUE',
       'Acohol_consumption', 'Mode_of_transit', 'Obesity_status']  # column modification


# In[14]:


df.columns


# # Outlier detection

# In[15]:


df.boxplot(column=['Daily_water_intake','TUE','Physical_exercise'])


# In[16]:


df.boxplot(column=['Age', 'Height', 'Weight', 'Feq_vegatable_consumption', 'No_main_meals'])


# ## these outliers are not handled because i might lose some vital information if removed, especially on age

# In[ ]:





# In[17]:


q1=df['Age'].quantile(0.25)
q3=df['Age'].quantile(0.75)
iqr=q3-q1
upperlimit=q3+(1.5*iqr)
lowerlimit=q1-(1.5*iqr)
upperlimit,lowerlimit


# In[18]:


df.loc[(df['Age']>upperlimit)]


# In[19]:


df['Mode_of_transit'].unique()


# # Exploratory Analysis

# In[20]:


df.shape


# In[21]:


dt=df.copy() # duplicating the dataframe


# In[22]:


dt.shape


# # 1) What is the relationship between age and obesity levels?

# In[23]:


df['Obesity_status'].unique()


# In[24]:


obesity_status_mapping={'Insufficient_Weight':0, 'Normal_Weight':1, 'Obesity_Type_I':2,
       'Obesity_Type_II':3, 'Obesity_Type_III':4,'Overweight_Level_I':5,
       'Overweight_Level_II':6}


# In[25]:


dt['Obesity_status']


# In[26]:


dt['Obesity_status']=dt['Obesity_status'].map(obesity_status_mapping)


# In[27]:


ob_level=dt.groupby('Obesity_status')['Age'].sum()


# In[28]:


ob_level.sort_values(ascending=False)


# # Hypothesis:
# ## H0:there is a relationship between age and obesity
# ## H1:there is no relationship

# In[29]:


dt.head()


# In[30]:


from sklearn.preprocessing import LabelEncoder


# In[31]:


dt['Obesity_status'].value_counts()


# In[32]:


le=LabelEncoder() # converting object values to numericals


# In[33]:


dt['Gender']=le.fit_transform(dt['Gender'])


# In[34]:


dt['family_history_with_overweight']=le.fit_transform(dt['family_history_with_overweight'])
dt['Feq_calory_food']=le.fit_transform(dt['Feq_calory_food'])


# In[35]:


df_corr=dt[['Age','Obesity_status']]


# In[36]:


df_corr.corr()


# In[37]:


dt.isnull().sum() #checking for null values


# In[38]:


dt['Obesity_status']


# In[39]:


from scipy.stats import pearsonr


# In[40]:


stat,p=pearsonr(dt['Age'],dt['Obesity_status'])

print(stat,p)

if p > 0.05:
    print ('there is  a relationship')
else:
    print('no relationship')


# ## there is no relationship  between age and obesity, therefore hypothesis is rejected

# In[ ]:





# # 2) Does obesity prevalence differ significantly by gender?

# ## hypothesis:
# ## H0: there is significance difference between obesity and gender
# ## H1: there is no significance difference between obesity and gender

# In[41]:


from scipy.stats import chi2_contingency


# In[42]:


contingency_table=pd.crosstab(dt['Gender'],dt['Obesity_status'])


# In[43]:


contingency_table


# In[44]:


stat,p,dof,expected=chi2_contingency(contingency_table)
print(stat,p)

if p > 0.05:
    print('there is difference')
else:
    print('no difference')


# ## obesity doesn't differ by gender, therefore hypothesis is rejected

# In[ ]:





# # 3) Are there specific age groups that have higher rates of obesity?

# In[45]:


ob_levels=dt.groupby('Obesity_status')['Age'].mean()


# In[46]:


ob_levels.head(20).sort_values(ascending=False)


# In[47]:


plt.figure(figsize=(15,5))
sns.histplot(x=dt['Age'],hue=dt['Obesity_status'],bins=150)
plt.title('obesity age range')
plt.xlabel('age range')
plt.ylabel('obesit status')


# ## The Age group of 28 has the highest level of obesity type 1, from age 20s, one is to watch their weight, there is high chance of adding more weight at that age range.

# In[ ]:





# # 4) How do eating habits (FAVC, FCVC, NCP, CAEC) influence obesity levels?

# ## Feq_vegatable_consumption

# In[48]:


veg_consumption=dt.groupby('Obesity_status')['Feq_vegatable_consumption'].mean()


# In[49]:


veg_consumption


# In[50]:


veg=dt.groupby('Obesity_status')[['Feq_vegatable_consumption','Weight']].mean()


# In[51]:


veg


# In[52]:


labels=['Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_I',
       'Obesity_Type_II', ' Obesity_Type_III', 'Overweight_Level_I',
       'Overweight_Level_II']


# In[53]:


plt.figure(figsize=(15,5))
sns.barplot(x=labels,y=veg_consumption,color='green')


# ## individuals with high weight gain comsumes lots of vegetable, other obesity levels consumes vegatable moderately

# In[54]:


dt['obe']=df['Obesity_status']


# In[55]:


dt[dt['Obesity_status']==3]['obe']


# ## Obesity_Type_III consumes high amount of vegetable

# In[ ]:





# # obesity and weight

# In[56]:


obesity=dt.groupby('Obesity_status')['Weight'].mean()


# In[57]:


label=['Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_I',
       'Obesity_Type_II', ' Obesity_Type_III', 'Overweight_Level_I',
       'Overweight_Level_II']


# In[58]:


obesity


# In[59]:


dt['Weight'].agg(['min','max'])


# In[60]:


dt[dt['Obesity_status']==2]['Weight'].mean() # from weight 92, your are obesed with some levels


# In[61]:


plt.figure(figsize=(10,5))
sns.barplot(x=label,y=obesity,palette='magma')
plt.title(' obesity and weight size')
plt.xlabel('obesity level')
plt.ylabel('weight level')


# In[62]:


dt['Weight'].mean() # average weight value


# ## individuals with  high weight value  from 92 to 120 are obesed, average weight is 86, anything above 86 is transitioning into overweight.

# In[ ]:





# # snacks in between meals

# In[63]:


obe_summary=dt.groupby('snacks')['Weight'].mean()


# In[64]:


obe_summary.head(30).sort_values(ascending=False)


# In[65]:


sns.barplot(obe_summary,palette='cividis')
plt.title('snacks in between meals')
plt.xlabel('obesity status')
plt.ylabel('caec')


# # consuming snacks always,sometimes leds to increase in weight gain, its best to take snacks frequently or not all

# In[ ]:





# In[66]:


dt.head()


# # family history of obesity

# In[67]:


family_history= dt.groupby(['Obesity_status'])['family_history_with_overweight'].mean()


# In[68]:


family_history


# In[69]:


yes_history=dt[dt['family_history_with_overweight']==1]['Obesity_status'].mean()
no_history=dt[dt['family_history_with_overweight']==0]['Obesity_status'].mean()


# In[70]:


history=[yes_history,no_history]


# In[71]:


plt.pie(history,labels=['yes','no'],autopct='%1.1f%%',colors=['red','green'])
plt.title('family history of obesity in %')


# ## 67% of the time, individuals with family history of obesity have high chance of obesity and overweight.
# ## 32% of the time, individuals with no family history of obesity tend to have obesity and overweight.

# In[72]:


df['family_history_with_overweight'].unique()


# In[73]:


las=['Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_I',
       'Obesity_Type_II', ' Obesity_Type_III', 'Overweight_Level_I',
       'Overweight_Level_II']


# In[74]:


plt.figure(figsize=(10,5))
sns.barplot(x=las,y=family_history,palette='coolwarm')
plt.title('family_history_of_overweight')


# ### individuals with family history of obesity have high chances of developing obesity with the average value of 3.3 compare to invdividual with no family history  of obesity with average value of 1.6.

# In[ ]:





# # freq calory foods

# In[75]:


calory_food=dt.groupby('Feq_calory_food')['Obesity_status'].mean()


# In[76]:


calory_food


# In[77]:


calory_foods=dt.groupby('Obesity_status')['Feq_calory_food'].mean()


# In[78]:


calory_foods


# In[79]:


la=['Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_I',
       'Obesity_Type_II', ' Obesity_Type_III', 'Overweight_Level_I',
       'Overweight_Level_II']


# In[80]:


plt.figure(figsize=(12,5))
sns.barplot(x=la,y=calory_foods,palette='viridis')
plt.title('how often calory food are consumed')
plt.xlabel('obesity statuts')
plt.ylabel('freq calory food')


# ## normal and insufficient weight individuals to overweight type 2 consume calory moderately, meanwhile overweight type 1 .
# ## to all obesity consumes calory food frequently.

# In[ ]:





# ## calory food and weight gain

# In[81]:


calory_foods=dt.groupby('Obesity_status')[['Feq_calory_food','Weight']].mean()


# In[82]:


calory_foods


# In[83]:


calory=dt.groupby('Obesity_status')['Feq_calory_food'].mean()


# In[84]:


calory


# In[85]:


value=['Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_I',
       'Obesity_Type_II', ' Obesity_Type_III', 'Overweight_Level_I',
       'Overweight_Level_II']
plt.figure(figsize=(10,10))
plt.bar(value,calory, color=['green','green','red','red','red','red','green'])
plt.title('calory consumption level')
plt.xlabel('obesity status')
plt.ylabel('level of calory consumption')


# In[ ]:





# ### individuals who consume high calory foods by 52% have obesity and gains more weight, its best  to consume less calory foods

# In[ ]:





# # Number of Meals

# In[86]:


meals=dt.groupby('Obesity_status')['No_main_meals'].mean()


# In[87]:


meals


# In[88]:


plt.pie(meals,autopct='%1.1f%%',labels=['0','1' ,'2 ','3 ','4','5','6'] )


# In[89]:


dt.groupby('Obesity_status')[['No_main_meals','Weight']].mean()


# In[90]:


values=['Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_I',
       'Obesity_Type_II', ' Obesity_Type_III', 'Overweight_Level_I',
       'Overweight_Level_II']
plt.figure(figsize=(10,5))
plt.bar(values,meals, color=['red','orange','green','yellow','red','blue','blue'])
plt.title('number of meals per obesity level')
plt.xlabel('obesity status')
plt.ylabel('average of meals')


# ## individuals with under weight consumes alot of food by 2.9 average value, individuals with high weight gain consume meals by 3.0 to 2.7 on average.
# ## the overweight individuals eats less daily.

# In[ ]:





# # Mode of transit and Physical exercise

# In[91]:


dt['Mode_of_transit']=le.fit_transform(dt['Mode_of_transit'])


# In[92]:


dt.groupby('Obesity_status')[['Physical_exercise','Mode_of_transit']].mean() # average values of physical exercise and mode of transit


# In[93]:


physical_ex=df.groupby('Obesity_status')['Physical_exercise'].mean()


# In[94]:


plt.figure(figsize=(10,10))
sns.barplot(physical_ex,palette='coolwarm')
plt.title('physical exercise engagement')
plt.xlabel('obesity status')
plt.ylabel('physical activity')


# ## those who are normal ,insufficient, and overweight_level_1 engage more in physical activities, obesity type is the least to engage in physical 
# ## activity.

# In[ ]:





# # Clustering

# In[ ]:





# # clustering features

# In[95]:


x=dt[['Gender',	'Age','Weight','Daily_water_intake','No_main_meals','family_history_with_overweight','Obesity_status']]


# In[96]:


x.head(2)


# In[97]:


from sklearn.cluster import KMeans


# In[98]:


from sklearn .preprocessing import MinMaxScaler


# In[99]:


scaler=MinMaxScaler(feature_range=(0,1)) # scalling


# In[100]:


x_scaled=scaler.fit_transform(x)


# ## determine the number of k using elbow method

# In[101]:


error=[]
for i in range(1,15):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(x_scaled)
    error.append( kmeans.inertia_ ) 


# In[102]:


plt.plot(range(1,15), error, marker='o',color='green')
plt.grid(True)
plt.savefig('k.jpg',pad_inches=0.8)


# # chose 3 as number of k because 3 is at the elbow point

# In[103]:


means=KMeans(3,random_state=0)


# In[104]:


means.fit(x_scaled) # fitting the values


# In[105]:


cluster=means.labels_ # getting the clusters


# In[106]:


centers=means.cluster_centers_ # the centriods


# In[107]:


centers


# In[108]:


dt['clusterer']=cluster


# In[109]:


dt.head()


# ## converting the object to numerical values

# In[110]:


dt['snacks']=le.fit_transform(dt['snacks'])
dt['Acohol_consumption']=le.fit_transform(dt['Acohol_consumption'])
dt['Calory_monitoring']=le.fit_transform(dt['Calory_monitoring'])
dt['Smoke']=le.fit_transform(dt['Smoke'])


# In[111]:


dt_drop=dt.drop('obe',axis=1,inplace=True)


# In[112]:


x['clus']=cluster


# In[113]:


x.head()


# In[114]:


dt.head()


# # cluster summary with average value

# In[115]:


summary=dt.groupby('clusterer').mean()


# In[116]:


summary


# In[117]:


x.head()


# # ploting the clusters

# In[118]:


df1=x[x['clus']==0]
df2=x[x['clus']==1]
df3=x[x['clus']==2]


# In[119]:


plt.scatter(df1['Age'],df1['Weight'],color='red')
plt.scatter(df2['Age'],df2['Weight'],color='yellow')
plt.scatter(df3.Age,df3.Weight,color='blue')
plt.title('clusters')
plt.xlabel('age')
plt.ylabel('weights')


# In[120]:


plt.scatter(x=x['Age'],y=x['Weight'],c=x['clus'],cmap='viridis')
plt.colorbar(label='Cluster') 
plt.title('clusters')
plt.xlabel('age')
plt.ylabel('weights')
plt.scatter(means.cluster_centers_[:,0],means.cluster_centers_[:,1],s=100,label='centroids',c='red')


# In[121]:


food_calory=dt.groupby('clusterer')['Feq_calory_food'].mean()


# In[122]:


food_calory


# In[123]:


plt.pie(food_calory, autopct='%1.1f%%',labels=['0','1','2'])
plt.title('Feq_calory_consumption')


# In[124]:


vegetable=dt.groupby('clusterer')['Feq_vegatable_consumption'].mean()


# In[125]:


vegetable


# In[126]:


plt.pie(vegetable,autopct='%1.1f%%',labels=['0','1','2'])
plt.title('Feq_vegatable_consumption')


# In[ ]:





# # kmeans clustering analysis

# In[ ]:





# # clustering findings:
# ## cluster 0:
# ### This group of individuals have average height of of 24.9 with 1.76 height,mostly males that has family history of obesity and weight of 94.2,
# ### consumes high calory food by 36.2%, eats less vegetable.
# ### consumes more snacks inbetween meals with no smoking, they have least attitude in monitoring their calory, exercises the most,
# ### they are mainly obesed.
# ## cluster 1:
# ### this group of individuals are mostly females, same average age as cluster 0,weight is 91.1 slightly less from cluster 0 by the difference of
# ### 3.048287000000002 ,they have family history of obesity and consumes high calory 35.1% but eats vegatable frequently in high quantity of 35%
# ### in foods ,consumes more water, monitor their calory and exercises very less, they are likely overweights.
# # cluster 2:
# ### this ggroup is a mix of males and females whose average age is 21.5,normal weight and no family history of obesity, takes less calory foods,
# ### consumes less vegetable,monitors calory in high extreme, exercises very well and consumes acohol in low qauntity. they are nornal weights.

# # clustering interpretation:
# ### Cluster 0: Older, male-dominant group with higher weight and moderate-to-high consumption habits, likely in the overweight category.
# ### Cluster 1: Female-dominant, similar in age to Cluster 0, slightly lower weight, and a similar profile in terms of consumption and exercise habits.
# ### Cluster 2: Younger, leaner group with a mix of genders, no family history of overweight, and healthier consumption habits with
# ### slightly more physical activity.

# In[ ]:





# # statistical analysis

# In[127]:


dt.describe()


# # Data Correlation

# In[128]:


dt_corr=dt.corr()


# In[129]:


dt_corr


# In[130]:


plt.figure(figsize=(15,10))
sns.heatmap(dt_corr,annot=True,vmin=-1,vmax=1)


# In[ ]:





# In[ ]:





# In[ ]:





# # model feature structuring

# In[131]:


x_df=dt[['Gender','Age','Weight','family_history_with_overweight','Feq_calory_food','Feq_vegatable_consumption',
         'snacks','Smoke','Daily_water_intake','Calory_monitoring','Physical_exercise']]
y_df=dt['Obesity_status']


# # splitting

# In[132]:


from sklearn .model_selection import train_test_split


# In[133]:


x_train,x_test,y_train,y_test=train_test_split(x_df,y_df,test_size=0.2,random_state=0)


# In[134]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# # model scaling

# In[135]:


x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# # model fitting and building

# In[136]:


from sklearn .neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn .naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn .svm import SVR


# In[137]:


logistic=LogisticRegression()
random_tree=RandomForestClassifier(n_estimators=200)
nb=MultinomialNB()


# In[138]:


sr=SVR()
tree=DecisionTreeClassifier(ccp_alpha=0.002,criterion='gini')


# In[139]:


sr.fit(x_train,y_train)
tree.fit(x_train,y_train)


# In[140]:


knn=KNeighborsClassifier(3)
knn.fit(x_train,y_train)


# In[141]:


logistic.fit(x_train,y_train)
random_tree.fit(x_train,y_train)
nb.fit(x_train,y_train)


# In[142]:


sr.score(x_train,y_train),tree.score(x_train,y_train)


# In[143]:


knn.score(x_train,y_train),logistic.score(x_train,y_train),random_tree.score(x_train,y_train),nb.score(x_train,y_train)


# In[144]:


ypred=knn.predict(x_test)


# In[145]:


tree_pred=tree.predict(x_test)


# In[146]:


randome_tree_pred=random_tree.predict(x_test)


# In[147]:


from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics


# In[148]:


metrics .accuracy_score(y_test,randome_tree_pred)


# In[149]:


metrics.accuracy_score(y_test,tree_pred)


# In[150]:


metrics.accuracy_score(y_test,ypred)


# ## Randomforest is chosen with high accuracy

# In[ ]:





# # model evaluation

# In[194]:


print(classification_report(y_test,randome_tree_pred))


# In[191]:


cm=confusion_matrix(y_test,randome_tree_pred)


# In[192]:


cm


# In[193]:


sns.heatmap(cm,annot=True)


# # model validation

# In[155]:


from sklearn.model_selection import cross_val_score,KFold


# In[195]:


kf=KFold(n_splits=5)
score=cross_val_score(random_tree,x_train,y_train,cv=kf)


# In[196]:


score


# In[158]:


## model is doing great at the validation state


# ## model testing

# In[197]:


random_tree.predict(scaler.transform([[0,21.0,64.0,1,0,2.0,2,0,2.0,0,0.0]]))


# In[199]:


random_tree.predict(scaler.transform([[0,	21.0,	56.0,	1,	0,	3.0	,2,	1,	3.0	,1,	3.0]]))


# In[198]:


random_tree.predict(scaler.transform([[0,	21.0,	64.0,	1,	0,	2.0,	2	,0,	2.0,	0,	0.0]]))


# In[200]:


random_tree.predict(scaler.transform([[1,	23.0,	77.0,	1,	0	,2.0,	2,	0,	2.0,	0,	2.0]]))


# In[201]:


random_tree.predict(scaler.transform([[1	,22.0	,89.8	,0	,0	,2.0,	2,	0,	2.0,	0	,0.0]]))


# In[202]:


random_tree.predict(scaler.transform([[1	,22.0	,30.0	,0	,0	,2.0,	2,	0,	2.0,	0	,0.0]]))


# In[203]:


random_tree.predict(scaler.transform([[1	,22.0	,76.0	,0	,0	,2.0,	2,	0,	2.0,	0	,0.0]]))


# In[168]:


## anything from weight 77 and above model predicts 6 which is over weights going to obesity


# In[169]:


importance=pd.DataFrame(tree.feature_importances_)


# In[170]:


importance


# # model saving

# In[175]:


import joblib


# In[176]:


model_obesity=joblib.dump(tree,'model_obesity.joblib')


# In[204]:


model_clustering=joblib.dump(means,'obesity_model_clustering.joblib') # clustering model


# In[ ]:




