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

from functions import functions

img=Image.open("my_app/logo.png")
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


st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("Dashboard 📊")
st.sidebar.markdown("Dashboard")

st.sidebar.markdown("### Filter Visual")
if data is not None:
    def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Try to convert datetimes into a standard format (datetime, no timezone)
        for col in df.columns:
            if is_object_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass

            if is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)

        modification_container = st.sidebar.container()

        with modification_container:
            to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
            for column in to_filter_columns:
                left, right = st.columns((1, 20))
                left.write("↳")
                # Treat columns with < 45 unique values as categorical
                if is_categorical_dtype(df[column]) or df[column].nunique() < 20:
                    user_cat_input = right.multiselect(
                        f"Values for {column}",
                        df[column].unique(),
                        default=list(df[column].unique()),
                    )
                    df = df[df[column].isin(user_cat_input)]
                elif is_numeric_dtype(df[column]):
                    _min = float(df[column].min())
                    _max = float(df[column].max())
                    step = (_max - _min) / 100
                    user_num_input = right.slider(
                        f"Values for {column}",
                        _min,
                        _max,
                        (_min, _max),
                        step=step,
                    )
                    df = df[df[column].between(*user_num_input)]
                elif is_datetime64_any_dtype(df[column]):
                    user_date_input = right.date_input(
                        f"Values for {column}",
                        value=(
                            df[column].min(),
                            df[column].max(),
                        ),
                    )
                    if len(user_date_input) == 2:
                        user_date_input = tuple(map(pd.to_datetime, user_date_input))
                        start_date, end_date = user_date_input
                        df = df.loc[df[column].between(start_date, end_date)]
                else:
                    user_text_input = right.text_input(
                        f"Substring or regex in {column}",
                    )
                    if user_text_input:
                        df = df[df[column].str.contains(user_text_input)]
        return df
    df=filter_dataframe(df)
    with st.expander("Select Columns To Visualize"):
        all_columns=df.columns
        st.markdown("#### ☑️Select Columns To Visualize Boxplot")
        b_target_cols=st.selectbox('Select target column',all_columns)
        b_column=st.selectbox('Selecct categorical column', all_columns)
        st.markdown("#### ☑️Select Columns To Visualize Bar Graph")
        bar_column=st.selectbox('select column for bar chart',all_columns)
        st.markdown("#### ☑️Select Columns To Visualize Map")
        lat=st.selectbox('Select latitude column',all_columns)
        long=st.selectbox('Select longitude column',all_columns)
        st.markdown("#### ☑️Select Columns To Visualize Mapbox")
        b_lat_column=st.selectbox('Select lat column',all_columns)
        b_lon_column=st.selectbox('Select lon column',all_columns)
        b_colors=st.selectbox('Select column for show color', all_columns)
        n=pd.DataFrame(df.loc[:,(df.dtypes==np.int64) | (df.dtypes==np.float)])
        all_columns_n = n.columns
    c1, c2=st.columns([1.9,2])
    
    with c1:
        if st.checkbox('Boxplot'):
            fig_bo = px.box(df, y = b_target_cols, color = b_column)
            c1.plotly_chart(fig_bo)	
    with c2:
        if st.checkbox('Bar Chart'):
            fig_ba = px.histogram(df, x = bar_column, color_discrete_sequence=['indianred'])
            c2.plotly_chart(fig_ba)


    
    df_m=pd.DataFrame({'lat':df[lat],'lon':df[long]})
    mask=df_m.notnull()
    df_m=df_m.where(mask).dropna()
    if st.checkbox('Map'):
        st.map(df_m)	

    
    c1, c2=st.columns([3,0.5])
    if st.checkbox('Mapbox'):
        px.set_mapbox_access_token('pk.eyJ1IjoiY2hob3J2eXZvdW4xMyIsImEiOiJjbDgxNHMwb2QwNTNmM3RvNDZhNnJka2tvIn0.2lWRwiJ9t-YYHV4vFQ3cGw') 
        with c2:
            plot=st.radio('Select layer bubble:', ( 'No size bubble','Add size bubble'))
        with c1:
            if plot=='No size bubble':
                df_m=pd.DataFrame({'latitude':df[b_lat_column],'longitude':df[b_lon_column], 'color':df[b_colors]})
                mask=df_m.notnull()
                df=df_m.where(mask).dropna()
                figMap = px.scatter_mapbox(df, lat='latitude', lon='longitude',color='color',
                                            color_continuous_scale=px.colors.cyclical.IceFire)
                st.plotly_chart(figMap, use_container_width=True)
    
            if plot=='Add size bubble':
                with c2:
                    bubble=st.selectbox('Select column for size bubble', all_columns_n)
                df_m=pd.DataFrame({'latitude':df[b_lat_column],'longitude':df[b_lon_column], 'color':df[b_colors],'bubble':df[bubble]})
                mask=df_m.notnull()
                df=df_m.where(mask).dropna()
                figMap = px.scatter_mapbox(df, lat='latitude', lon='longitude',color='color', size='bubble',
                                            color_continuous_scale=px.colors.cyclical.IceFire)
                st.plotly_chart(figMap, use_container_width=True)
            
            
       
	


    
    
	    
	
    
    




	


    
    
	    
	
    
    



