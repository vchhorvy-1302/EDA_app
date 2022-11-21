import base64

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu

img=Image.open("my_app/logo.png")
st.set_page_config(page_title="Z1App/EDA/About_us",page_icon=img,layout="wide")


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
#End Theme

st.markdown(page_bg_img, unsafe_allow_html=True)
image = Image.open('my_app/logo.png')

st.image(image, width=120)
st.sidebar.markdown("Contents")
st.sidebar.success("Select a demo above.")
st.write("# Welcome to Z1 Data Exploration Web Aplication 👋")
col1, col2 = st.columns([1.5,3])
with col1:
    st.markdown("### **Don't Wanna Spend Much Time!**")
    st.markdown("This Web Application can help you through your data by yourselve without code require and spend less time. If you used to spend a weeks to  go through your data by code to understand it before cleaning or analyzing.Drag and drop your data, get through all the nesessaries informations that you need to know before cleaning or analyzing insight your data.We believe that our workflow make your data more understandable.")
with col2:
    st.image('https://i.pinimg.com/originals/cf/94/7b/cf947b46283c10c47e3d5d945afb7053.gif', width=650)
    


text_style = '<p style="font-family:sans-serif; color:Purple; font-size: 20px;">Drop your dataset here to understand your data more:(csv is recommended, it is faster) 👇'
st.markdown(text_style,  unsafe_allow_html=True)


file_format = st.radio('Select file format:', ('csv', 'excel'), key='file_format')
data = st.file_uploader(label = '')

use_defo = st.checkbox('Use example Dataset')
if use_defo:
    data = 'sample_dataset.csv'


if data:
    if file_format == 'csv' or use_defo:
        df = pd.read_csv(data)
    else:
        df = pd.read_excel(data)
    
