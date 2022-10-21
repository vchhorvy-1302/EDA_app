import base64
from lib2to3.pgen2.pgen import DFAState

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pandas.api.types import (is_categorical_dtype, is_datetime64_any_dtype,
                              is_numeric_dtype, is_object_dtype)
from PIL import Image
from streamlit_option_menu import option_menu

from functions import functions

img=Image.open("logo.png")
st.set_page_config(page_title ="Data Dashboard",page_icon = "Active",layout = "wide",
)
#Theme
df = px.data.iris()
@st.experimental_memo
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://static.vecteezy.com/system/resources/previews/006/861/161/original/light-blue-background-gradient-wall-design-vector.jpg");
background-size: cover;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""


st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("Dashboard 📊")
st.sidebar.markdown("Dashboard")



with st.expander(""):
	file_format = st.radio('Select file format:', ('csv', 'excel'), key='file_format')
	data = st.file_uploader(label = '')

	use_defo = st.checkbox('Use example Dataset')
	if use_defo:
		data = 'scrap_data_sample.csv'


	if data:
		if file_format == 'csv' or use_defo:
			df = pd.read_csv(data)
		else:
			df = pd.read_excel(data)



if data is not None:
	c1, c2=st.columns([1.9,2])
	all_columns=df.columns
	
	with c1:
		all_columns=df.columns
		st.sidebar.markdown('### ☑️Select for box plot')
		target_cols=st.sidebar.selectbox('Select target column',all_columns)
		cat_column=st.sidebar.selectbox('Selecct categorical column', all_columns)
		if st.checkbox('Boxplot'):
			fig_bo = px.box(df, y = target_cols, color = cat_column)
			c1.plotly_chart(fig_bo)	
		
	with c2:
		st.sidebar.markdown('### ☑️Select for bar graph')
		select_column=st.sidebar.selectbox('select column for bar chart',all_columns)
		if st.checkbox('Bar Chart'):
			fig_ba = px.histogram(df, x = select_column, color_discrete_sequence=['indianred'])
			c2.plotly_chart(fig_ba)

if data is not None:
	st.sidebar.markdown('### ☑️Select columns for plot map')
	lat=st.sidebar.selectbox('Select latitude column',all_columns)
	long=st.sidebar.selectbox('Select longitude column',all_columns)
	df=pd.DataFrame({'lat':df[lat],'lon':df[long]})
	mask=df.notnull()
	df=df.where(mask).dropna()
	if st.checkbox('Map'):
		st.map(df)	
		
	


    
    
	    
	
    
    



