import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import plotly.express as px

data = pd.read_csv('matches.csv')
data2 = pd.read_csv('matches2.csv')
data3 = pd.read_csv('matches3.csv')

# Gabungkan DataFrame utama dengan DataFrame baru
data = pd.concat([data, data2, data3], ignore_index=True)

def predict_outcome(home_team, away_team) :
    data[ "date" ] = pd.to_datetime(data[ "date" ])
    data[ "result" ] = data[ "result" ].replace({"W" : "Win", "L" : "Lose", "D" : "Draw"})
    data[ "target" ] = data[ "result" ]
    data[ "venue_code" ] = data[ "venue" ].astype("category").cat.codes
    data[ "opp_code" ] = data[ "opponent" ].astype("category").cat.codes
    data[ "hour" ] = data[ "time" ].str.replace(":.+", "", regex=True).astype("int")
    data[ "day_code" ] = data[ "date" ].dt.dayofweek

    predictors = [ "venue_code", "opp_code", "hour", "day_code" ]

    train = data[ data[ "date" ] < '2022-01-01' ]

    rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

    rf.fit(train[ predictors ], train[ "target" ])

    latest_match_home = \
    data[ (data[ "team" ] == home_team) | (data[ "opponent" ] == home_team) ].sort_values("date").iloc[ -1 ]
    latest_match_away = \
    data[ (data[ "team" ] == away_team) | (data[ "opponent" ] == away_team) ].sort_values("date").iloc[ -1 ]

    prediction_data = pd.DataFrame({
        "venue_code" : latest_match_home[ "venue_code" ],
        "opp_code" : latest_match_away[ "opp_code" ],
        "hour" : latest_match_home[ "hour" ],
        "day_code" : latest_match_home[ "day_code" ]
    }, index=[ 0 ])

    predicted_outcome = rf.predict(prediction_data)[ 0 ]
    predicted_probabilities = rf.predict_proba(prediction_data)[ 0 ]

    return predicted_outcome, dict(zip(rf.classes_, predicted_probabilities))

st.title('Football Analytics')
menu = st.sidebar.selectbox('Menu', [ 'Home', 'Data Showcase & Predictors'])

if menu == 'Home' :
    img = st.image('Bola.jpg')
    st.write('Ami Rofiatin                  223307032')
    st.write('Ersado Cahya Buana            223307039')
    st.write('Muhammad Fariz                223307049')
    st.write('Sang Pralambang Sri Hendra    223307057')

elif menu == 'Data Showcase & Predictors' :
    st.header('Data Showcase')

    st.subheader('Entire Data')
    data

    st.subheader('Grafik Jumlah Gol Tim Tuan Rumah dan Tamu')
    fig = px.bar(data.groupby('team')[['gf', 'ga']].sum().reset_index(), x='team', y=['gf', 'ga'],
                 labels={'value': 'Goals', 'variable': 'Metric'}, barmode='group',
                 title='Jumlah Gol Tim Tuan Rumah dan Tamu')
    st.plotly_chart(fig)

    st.subheader('Distribusi Hasil Pertandingan')
    fig, ax = plt.subplots( )
    data[ 'result' ].value_counts( ).plot(kind='bar', ax=ax)
    st.pyplot(fig)

    st.subheader('Rata-rata Jumlah Gol per Pertandingan Seiring Waktu')
    fig = px.area(data.groupby('date')['gf'].mean().reset_index(), x='date', y='gf',
                  labels={'gf': 'Average Goals', 'date': 'Date'}, title='Rata-rata Jumlah Gol per Pertandingan Seiring Waktu')
    st.plotly_chart(fig)

    st.write('')
    st.header('Prediction')
    st.write("Pilih nama tim tuan rumah dan tamu untuk mendapatkan prediksi hasil pertandingan.")

    home_team = st.selectbox("Nama Tim Tuan Rumah", data[ "team" ].unique( ), index=None,placeholder='Pilih Team')
    away_team = st.selectbox("Nama Tim Tamu", data[ "team" ].unique( ),index=None, placeholder='Pilih Team')

    if st.button("Prediksi") :
        predicted_outcome, predicted_probabilities = predict_outcome(home_team, away_team)
        st.write(f"Prediksi hasil pertandingan antara Tuan Rumah {home_team} vs {away_team}: {predicted_outcome}")
        st.write("Probabilitas:")
        for outcome, probability in predicted_probabilities.items( ) :
            st.write(f"{outcome}: {probability:.2%}")
