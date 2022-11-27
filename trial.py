
import streamlit as st

import pandas as pd

import numpy as np

import pickle

import base64

from PIL import Image

st.set_page_config(layout="wide")



st.title("""

    App for prediction of temperature of EV motor windings

   

    """)

col1, col2,col3 = st.columns((1.5,0.1,1.5))



    
with col1:
    

    
    u_q	= st.number_input('q_component of voltage [V]:',min_value=0, max_value=135) 
    coolant	= st.number_input('Coolant outlet temperature [Deg]:',min_value=10, max_value=105)

    u_d	= st.number_input('d_component of voltage[V]:',min_value=-135, max_value=135)

    motor_speed	= st.number_input('Motor speed [RPM]:',min_value=-275, max_value=6000)

    i_d	= st.number_input('d_component of current[A]:',min_value=-270, max_value=0)

   

    ambient		= st.number_input('Ambient temperature [Deg]:',min_value=15, max_value=30)

    torque = st.number_input('Torque [Nm]:',min_value=-246, max_value=246)

    model = pickle.load(open(r'GBoost_web.pickle','rb'))
    



with col3:
    # Add chart #1
    st.subheader("""

    

    This Machine learning App can be used for predicting temperature of motor of Electric vechicles based on parameters like torque, speed ..etc
    

    """)
    
    
    if st.button('Motor winding temperature',help='Click me after entering parameters on left pane'):
        Temp1 = model.predict([[u_q,coolant, u_d, motor_speed,i_d, ambient, torque]] )
        st.success(f'The Temperature of winding is {Temp1[0]:.1f} Deg')			
    

   
    image = Image.open('Motor1.JPG')
    #image = image.resize((2, 2), resample=Image.BILINEAR)
    #st.image(image,use_column_width='auto',width=0.5)
    st.image(image,width=1,use_column_width='always')
    

    #st.image(image, caption=None, width=2, use_column_width='auto', clamp=False, channels="RGB", output_format="auto")


    st.write(""" 
    Reference: https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature

    """)
