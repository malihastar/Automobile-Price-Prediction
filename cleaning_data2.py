
"""
Created on Fri Oct 29 16:31:59 2021

@author: lenovo
"""
import pandas as pd
import numpy as np
import datetime

df=pd.read_csv('voks.csv')
# Replace turkish date to english


# Select necessary columns 
df=df[['Seri','Model','Yıl','Km','Vites','Yakıt','Fiyat']]
df['Fiyat'].dtype
# remove TL string from Fiyat columns
df['Fiyat']=df['Fiyat'].str.strip("(TL)")
df=df.astype({"Fiyat":float})

df.isnull().sum()
# drop null values from dataset. the dataset has just 2 null values 
df=df.dropna()
# convert yıl columns to datetime
df['Yıl'] = pd.to_datetime(df['Yıl'], format="%Y")
df['Yıl']=df['Yıl'].dt.year
df['Yıl'].dtype

df.corr()

# remove outliers from Fiyat columns:

fiyat_df=df['Fiyat']
# Q1 = fiyat_df.quantile(0.25)
# Q3 = fiyat_df.quantile(0.75)
# IQR = Q3-Q1
# print(Q1,Q3,IQR)
# alt_sinir = Q1- 1.5*IQR
# ust_sinir = Q3 + 1.5*IQR
# print(alt_sinir,ust_sinir)
# aykiri=(fiyat_df < alt_sinir) | (fiyat_df > ust_sinir)
# len(fiyat_df[aykiri])
# fiyat_df[aykiri]=ust_sinir
# plot fiyat columns
import seaborn as sns
sns.boxplot(x = fiyat_df);
sns.boxplot(x=df['Km'])

#Removes outliers from Km columns:

# km_df=df['Km']

# Q1 = km_df.quantile(0.25)
# Q3 = km_df.quantile(0.75)
# IQR = Q3-Q1
# Q1,Q3,IQR
# alt_sinir2 = Q1- 1.5*IQR
# ust_sinir2 = Q3 + 1.5*IQR
# print(alt_sinir2,ust_sinir2)
# aykiri2=(km_df < alt_sinir2) | (km_df > ust_sinir2)
# len(km_df[aykiri2])
# # sns.boxplot(x = km_df);
# km_df[aykiri2]=ust_sinir2
# # sns.boxplot(x = km_df)

# # Dummy Encoding  for many columns 

# df.Yakıt.value_counts().sort_values(ascending=False)
# df = pd.get_dummies(df, columns = ["Yakıt"], prefix = ["Yakıt"])
# df.drop('Yakıt_Benzin & LPG', inplace=True, axis=1)
# df=pd.get_dummies(df, columns = ["Vites"], prefix = ["Vites"])
# df.drop('Vites_Yarı Otomatik', inplace=True, axis=1)

# df=pd.get_dummies(df, columns = ["Model"],drop_first = True)
# df=pd.get_dummies(df, columns = ["Seri"],drop_first = True)


# df.to_csv('Data2.csv')




