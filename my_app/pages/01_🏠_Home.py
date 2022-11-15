import base64

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

img=Image.open("my_app/logo.png")
st.set_page_config(page_title="Z1App/EDA/Home",page_icon=img)

	

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
[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;


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

	#End Theme


image = Image.open('my_app/logo.png')
st.sidebar.image(image,width=100)
st.title("Home Page 🏠")
st.header('Exploratory Data Analysis:EDA')
with st.expander(""):
	file_format = st.radio('Select file format:', ('csv', 'excel'), key='file_format')
	data = st.file_uploader(label = '')

	use_defo = st.checkbox('Use example Dataset')
	if use_defo:
		data = 'my_app/scrap_data_sample.csv'


	if data:
		if file_format == 'csv' or use_defo:
			df = pd.read_csv(data)
		else:
			df = pd.read_excel(data)
 
st.subheader('Overview')
st.markdown('''This is the project of data exploration based on our company method implement. Every first step in any data-intensive
			project is understanding the available data.To this end, data scientists or data engineer spend a significant
			part of their time carrying out data quality assessments and data exploration. In spite of this being a crucial
			step, it usually requires repeating a series of menial tasks before the data scientist gains an understanding of
			the dataset and can progress to the next steps in the project.In this talk, it will detail the inner workings of 
			workflow and important parts that it will be built which automates this drudge work, enables efficient data exploration,
			and kickstarts data science projects.''')


st.markdown("### EDA Process")
st.image('https://www.researchgate.net/profile/Cristina-Sousa-2/publication/342282008/figure/fig2/AS:903839578353665@1592503556728/Exploratory-Data-Analysis-EDA-steps.png', width=700)

