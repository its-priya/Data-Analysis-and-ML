import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Loading trained Model
model= open('KNN_Classifier.pkl','rb')
knn_model= joblib.load(model)

# Loading images
setosa= Image.open('irisSetosa.jpeg')
versicolor= Image.open('irisVersicolor.jpeg')
virginica= Image.open('irisVirginica.jpg')

st.title('Iris Flower Classification')
st.sidebar.title('Features')

# Initializing parameters
parameters_list= ['Sepal Length (cm)', 'Sepal Width (cm)',
'Petal Length (cm)', 'Petal Width (cm)']
parameters= []
parameters_default= ['5.2', '3.2', '4.2', '1.2']
values= []

for pm, pm_def in zip(parameters_list, parameters_default):
    values= st.sidebar.slider(label= pm, key= pm, 
    min_value= 0.0, max_value= 8.0, value= float(pm_def), step= 0.1)
    parameters.append(values)

input_vars= pd.DataFrame([parameters], columns= parameters_list, dtype=float)
st.write('\n\n')

# Display predicted Result
if st.sidebar.button('Click here to Classify'):
    prediction= knn_model.predict(input_vars)
    if prediction==0:
        st.subheader('Iris Setosa')
        st.write('\n')
        st.image(setosa, width=450)
    elif prediction==1:
        st.subheader('Iris Versicolor')
        st.write('\n')
        st.image(versicolor, width=450)
    else:
        st.subheader('Iris Virginica')
        st.write('\n')
        st.image(virginica, width=450) 

