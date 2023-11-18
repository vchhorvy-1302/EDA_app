import base64
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

df = px.data.iris()
@st.cache_data
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
st.title('Basic data cleaning')
with st.expander(""):
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

# Drop columns

if data is not None:
    df = df.copy()
    all_column=df.columns
    st.markdown('### Remove The Unwanted Columns')
    all_c=st.multiselect("Select Columns To Drop", all_column)
    if st.checkbox("Remove selected columns"):
        df.drop(all_c, axis=1, inplace=True)
        st.write(df)
    if st.checkbox('Remove all empty columns'):
        
        df.dropna(axis=1,how='all', inplace=True)
        st.write(df)
        st.write(df.shape)
    st.markdown('### Remove The Empty Rows ')
    
    cols=df.columns
    all_co=st.multiselect("Select Columns To Remove Row", cols)
    if st.checkbox("Remove selected columns's rows "):
        df.dropna(subset=all_co, inplace=True)
        st.write(df)
        st.write(df.shape)
    if st.checkbox("Remove Empty Rows") :
        df.dropna(inplace = True, how="all")
        st.write(df)
        st.write(df.shape)

    st.markdown('### Remove Outliers')
    all_cols=df.columns
    cols=st.multiselect("Select Outlier's Columns", all_cols)
    Q1 = df[cols].quantile(0.25)
    Q3 = df[cols].quantile(0.75)
    IQR = Q3 - Q1
    
    if st.checkbox('Remove all outliers'):
        df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
        st.write(df)
        st.write(df.shape)
    if st.checkbox('Remove extreme value'):
        df = df[~((df[cols] < (Q1 - 3 * IQR)) |(df[cols] > (Q3 + 3 * IQR))).any(axis=1)]
        st.write(df)
        st.write(df.shape)
    
    st.markdown("### Remove Duplicate")
    if st.checkbox('Remove Duplicate Base on Columns'):
        df.drop_duplicates(subset=cols, inplace=True)
        st.write(df)
        st.write(df.shape)
    if st.checkbox("Remove Duplicate Rows"):
        df.drop_duplicates(inplace=True)
        st.write(df)
        st.write(df.shape)
    
        
    
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    csv= convert_df(df)
    
    st.download_button(
        label="Download cleaned data as CSV",
        data=csv,
        file_name='df.csv',
        mime='text/csv',
    )


 

         
    
    
      
    
