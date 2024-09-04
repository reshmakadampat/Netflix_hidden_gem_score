import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('netflix-rotten-tomatoes-metacritic-imdb.csv')
df = df.drop(columns=['Title','Netflix Link','IMDb Link','Image','Poster','TMDb Trailer','Trailer Site'],axis=1)
df = df.drop(columns=['Boxoffice','Metacritic Score','Production House','Awards Received'],axis=1)
freqgrph=df.select_dtypes(include=['float','int'])
for col in freqgrph.columns:
       df[col]=df[col].fillna(df[col].median())
for col in df.select_dtypes(include=['object']):
       df[col]=df[col].fillna(df[col].mode()[0])
large_unique_col=['Tags','Country Availability','Director','Writer','Actors','Summary','IMDb Votes']
from sklearn.impute import SimpleImputer

# Replace rare categories with the most frequent category
for col in large_unique_col:
   imputer = SimpleImputer(strategy='most_frequent')
   df[col] = imputer.fit_transform(df[[col]]).ravel()
data=df.copy()
status_mapping = {
    'Series': 0,
    'Movie': 1,
}
data['Series or Movie']=data['Series or Movie'].map(status_mapping)
runtime_mapping = {
    '< 30 minutes': 0,
    '30-60 mins': 1,
    '1-2 hour': 2,
    '> 2 hrs': 3
}
data['Runtime'] = data['Runtime'].map(runtime_mapping)
data.drop(columns=['Summary'],axis=1,inplace=True)
data.drop(columns=['Tags'],axis=1,inplace=True)
data.drop(columns=['Actors'],axis=1,inplace=True)
categorical_col=['Genre','Languages','Country Availability','Director','Writer','View Rating']
for col in categorical_col:
   frequency_encoding = data[col].value_counts()
   data[col] = data[col].map(frequency_encoding)
data['Release Date'] = pd.to_datetime(data['Release Date'], format='%d %b %Y')
data['Release Date Day'] = data['Release Date'].dt.day
data['Release Date Month'] = data['Release Date'].dt.month
data['Release Date Year'] = data['Release Date'].dt.year
data.drop(columns=['Release Date'],axis=1,inplace=True)
data['Netflix Release Date'] = pd.to_datetime(data['Netflix Release Date'], format='%Y-%m-%d')
data['Netflix Release Date Day'] = data['Netflix Release Date'].dt.day
data['Netflix Release Date Month'] = data['Netflix Release Date'].dt.month
data['Netflix Release Date Year'] = data['Netflix Release Date'].dt.year
data.drop(columns=['Netflix Release Date'],axis=1,inplace=True)
x=data.drop(columns=['Hidden Gem Score'])
y=data['Hidden Gem Score']
x1=x.copy()
x1.drop(['Genre', 'Languages', 'Series or Movie', 'Country Availability',
       'Runtime', 'Director', 'Writer', 'View Rating', 'Release Date Day',
       'Release Date Month', 'Release Date Year', 'Netflix Release Date Day',
       'Netflix Release Date Month', 'Netflix Release Date Year'],axis=1,inplace=True)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x1=scaler.fit_transform(x1)
x1=pd.DataFrame(x1,columns=['IMDb Score', 'Rotten Tomatoes Score', 'Awards Nominated For',
       'IMDb Votes'])
x2=x.drop(['IMDb Score', 'Rotten Tomatoes Score', 'Awards Nominated For',
       'IMDb Votes'],axis=1)
x=pd.concat([x1,x2],axis=1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
import xgboost as xgb
simple_reg = xgb.XGBRegressor(
    objective="reg:squarederror",
    seed=123,
    learning_rate=0.1
)
#objective="reg:squarederror" to minimize the squared error
#Seeds=123 ensure that the results will be the same if the code is run again with the same data.
#model’s speed at learning from mistakes.learning_rate=0.1 it can prevent the model from making big mistakes by overreacting to errors.
simple_reg.fit(x_train,y_train)

y_pred1 = simple_reg.predict(x_test)
import pickle
with open('netflix_prediction.pkl','wb') as model_file:
  pickle.dump(simple_reg,model_file)