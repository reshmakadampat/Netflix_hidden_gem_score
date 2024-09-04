from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd
app=Flask(__name__)
with open('netflix_prediction.pkl','rb') as model_files:
  netflix=pickle.load(model_files)
@app.route('/')
def home():
  return render_template('details.html')
@app.route('/read' ,methods=['POST'])
def model():
  Genre=request.form['Genre']
  Languages=request.form['Languages']
  Series_or_Movie=request.form['Series or Movie']
  Country_Availability=request.form['Country Availability']
  Runtime=(request.form['Runtime'])
  Director=request.form['Director']
  Writer=request.form['Writer']
  View_Rating=request.form['View Rating']
  IMDb_Score=float(request.form['IMDb Score'])
  Rotten_Tomatoes_Score=float(request.form['Rotten Tomatoes Score'])
  Awards_Nominated_For=int(request.form['Awards Nominated For'])
  IMDb_Votes=int(request.form['IMDb Votes'])
  Release_Date=request.form['Release Date']
  Netflix_Release_Date=request.form['Netflix Release Date']

  data=pd.DataFrame({
    'IMDb_Score':[IMDb_Score],
    'Rotten_Tomatoes_Score':[Rotten_Tomatoes_Score],
    'Awards_Nominated_For':[Awards_Nominated_For],
    'IMDb_Votes':[IMDb_Votes]
  })

  map=pd.DataFrame({
    'Series_or_Movie':[Series_or_Movie],
    'Runtime':[Runtime]
  })
  status_mapping = {
    'Series': 0,
    'Movie': 1,
  }
  map['Series_or_Movie']=map['Series_or_Movie'].map(status_mapping)

  runtime_mapping = {
    '< 30 minutes': 0,
    '30-60 mins': 1,
    '1-2 hour': 2,
    '> 2 hrs': 3
  }
  map['Runtime'] = map['Runtime'].map(runtime_mapping)

  df=pd.DataFrame({
    'Genre':[Genre],
    'Languages':[Languages],
    'Country Availability':[Country_Availability],
    'Director':[Director],
    'Writer':[Writer],
    'View Rating':[View_Rating]
  })
  for col in df.columns:
   frequency_encoding = df[col].value_counts()
   df[col] = df[col].map(frequency_encoding)
  df.insert(2, 'Series or Movie',map['Series_or_Movie'] )
  df.insert(4, 'Runtime',map['Runtime'] )
  time=pd.DataFrame({
     'Release_Date':[Release_Date],
    'Netflix_Release_Date':[Netflix_Release_Date]
   })
  time['Release_Date'] = pd.to_datetime(time['Release_Date'], format='%d %b %Y')
  time['Release Date Day'] = time['Release_Date'].dt.day
  time['Release Date Month'] = time['Release_Date'].dt.month
  time['Release Date Year'] = time['Release_Date'].dt.year
  time.drop(columns=['Release_Date'],axis=1,inplace=True)
  time['Netflix_Release_Date'] = pd.to_datetime(time['Netflix_Release_Date'], format='%Y-%m-%d')
  time['Netflix Release Date Day'] = time['Netflix_Release_Date'].dt.day
  time['Netflix Release Date Month'] = time['Netflix_Release_Date'].dt.month
  time['Netflix Release Date Year'] = time['Netflix_Release_Date'].dt.year
  time.drop(columns=['Netflix_Release_Date'],axis=1,inplace=True)
  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler()
  data_scaled = scaler.fit_transform(data)
  input_df = pd.DataFrame(data_scaled, columns=['IMDb Score', 'Rotten Tomatoes Score', 'Awards Nominated For', 'IMDb Votes'])
  x=pd.concat([input_df,df],axis=1)
  final=pd.concat([x,time],axis=1)
  prediction=netflix.predict(final)
  return render_template('details.html',prediction_result=prediction[0])

if __name__=='__main__':
  app.run(debug=True)