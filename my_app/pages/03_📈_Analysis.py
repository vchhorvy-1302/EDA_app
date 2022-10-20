
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

matplotlib.use("Agg")
import plotly.figure_factory as ff
import seaborn as sns

from functions import functions

img=Image.open("C:\my_app/image/logo.png")
st.set_page_config(page_title="Z1App/EDA/DataAnalysis", page_icon=img)


#Theme
df = px.data.iris()
@st.experimental_memo
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("C:\my_app\image\sidebar.png")

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


st.title("📈 Analysis")
# hide uploader file
with st.expander(""):
	file_format = st.radio('Select file format:', ('csv', 'excel'), key='file_format')
	data = st.file_uploader(label = '')

	use_defo = st.checkbox('Use example Dataset')
	if use_defo:
		data = 'c:/my_app/sample_dataset/scrap_data_sample.csv'


	if data:
		if file_format == 'csv' or use_defo:
			df = pd.read_csv(data)
		else:
			df = pd.read_excel(data)

activity = ["Quantitative","Qualitative"]
with st.sidebar:

    st.header("ANALYSIS WITH OPTION MENU")

    selected = option_menu(
    menu_title=None,  # required
    options=activity,  # required
    icons=None,  # optional
    menu_icon="menu-down",  # optional
    default_index=0,  # optional
    )

    # Qualitative
if selected=="Qualitative":
    st.markdown("#### Qualitative Analysis")
    st.markdown("##### 1️⃣ Categorical Data and Boolean Data")
    if data is not None:
        c=pd.DataFrame(df.loc[:,df.dtypes==np.object ])
        b=pd.DataFrame(df.select_dtypes(include=['bool']))
        f=pd.concat([c, b], axis=1)

        all_columns_c = pd.concat([c, b], axis=1).columns
        type_of_plot = st.selectbox("Select Type of Plot",["bar","pie"])
        if type_of_plot=='bar':
            if len(all_columns_c) == 0:
                st.write('There is no categorical columns in the data.')
            else:
                selected_cat_cols = functions.multiselect_container('Choose columns for Count plots:', all_columns_c, 'Count')
                st.markdown('###### Count plots of categorical columns')
                i = 0
                while (i < len(selected_cat_cols)):
                    c1, c2 = st.columns(2)
                    for j in [c1, c2]:

                        if (i >= len(selected_cat_cols)):
                            break

                        fig = px.histogram(df, x = selected_cat_cols[i], color_discrete_sequence=['indianred'])
                        j.plotly_chart(fig)
                        i += 1
        if type_of_plot=='pie':
            if len(all_columns_c) == 0:
                st.write('There is no categorical columns in the data.')
            else:
                selected_cat_cols = functions.multiselect_container('Choose columns for Count plots:', all_columns_c, 'Count')
                st.markdown('###### Count plots of categorical columns')
                i = 0
                while (i < len(selected_cat_cols)):
                    c1, c2 = st.columns(2)
                    for j in [c1, c2]:

                        if (i >= len(selected_cat_cols)):
                            break
                        st.write(f[selected_cat_cols[i]].value_counts().plot.pie(autopct="%1.1f%%"))
                        plt.style.use('classic')
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        st.pyplot()
                        i +=1

    st.markdown("##### 2️⃣ Date Data")
    if data is not None:
        d=pd.DataFrame(df.select_dtypes(include=['datetime64']))
        d_columns=pd.DataFrame(df.select_dtypes(include=['datetime64'])).columns
        if len(d_columns) == 0:
            st.write('There is no numerical columns in the data.')
        else:
            all_columns_names = d.columns.tolist()
            selected_columns_names = st.multiselect("Select Categorical Columns To Plot",all_columns_names)
            type_of_plot = st.selectbox("Select Type of Plot",["bar","line"])
            Date=st.selectbox("Select Type Of Date",["Year","Month"])
        if st.button("Click To Plot"):
            st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names, Date))
            if type_of_plot=="bar" and Date=="Year":
                d["da"]=d[selected_columns_names].dt.year()
                st.write(d["da"].iloc[:,-1].value_counts().plot(kind='bar'))
                plt.style.use('classic')
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()
            if type_of_plot=="bar" and Date=="Month":
                f["da"]=d[selected_columns_names].dt.month()
                st.write(f["da"].iloc[:,-1].value_counts().plot(kind='bar'))
                plt.style.use('classic')
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()
            if type_of_plot=="line" and Date=="Year":
                d["da"]=d[selected_columns_names].dt.year()
                cust_data = d["da"]
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.line_chart(cust_data)
            if type_of_plot=="line" and Date=="Month":
                d["da"]=d[selected_columns_names].dt.month()
                cust_data = d["da"]
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.line_chart(cust_data)
            

            





            
# Quantitative
if selected=="Quantitative":
    st.markdown("#### Quantitative Analysis")
    st.markdown("##### 1️⃣ Numerical Data")

    
#Univariate
    st.markdown("##### 📌Univariate Analysis")
    if data is not None:
        n=pd.DataFrame(df.loc[:,(df.dtypes==np.int64) | (df.dtypes==np.float)])
        all_columns_n = n.columns
        type_of_plot = st.selectbox("Select Type of Plot",["box","hist"])
        if type_of_plot=="box":
            if len(all_columns_n) == 0:
                st.write('There is no numerical columns in the data.')
            else:
                selected_num_cols = functions.multiselect_container('Choose columns for Box plots:', all_columns_n, 'Box')
                st.markdown('Box plots')
                i = 0
                while (i < len(selected_num_cols)):
                    c1, c2 = st.columns(2)
                    for j in [c1, c2]:
                        
                        if (i >= len(selected_num_cols)):
                            break
                        
                        fig = px.box(df, y= selected_num_cols[i])
                        j.plotly_chart(fig, use_container_width = True)
                        i += 1
        if type_of_plot=="hist":
            if len(all_columns_n) == 0:
                st.write('There is no numerical columns in the data.')
            else:
                selected_num_cols = functions.multiselect_container('Choose columns for Hist plots:', all_columns_n, 'Hist')
                st.markdown('Hist plots')
                i = 0
                while (i < len(selected_num_cols)):
                    c1, c2 = st.columns(2)
                    for j in [c1, c2]:
                        
                        if (i >= len(selected_num_cols)):
                            break
                        
                        fig = px.histogram(df, x = selected_num_cols[i])
                        j.plotly_chart(fig, use_container_width = True)
                        i += 1
# Bivariate
    st.markdown('##### 📌Bivariate Analysis')
    text_style = '<p style="font-family:sans-serif; color:Blue; font-size: 15px;">Correlation between numerical and numcerial👇'
    st.markdown(text_style,unsafe_allow_html=True)
    if data is not None: 
        all_columns=n.columns                  
        selected_num_cols =st.selectbox('Choose columns for Target Value:', all_columns)  
        num_cols = st.selectbox("Select target column:", n.columns, index = len(n.columns) - 1) 
        if st.button('Generate scatter plot'):   
            fig = px.scatter(df, x= selected_num_cols, y = num_cols)
            st.plotly_chart(fig)

    text_style_2 = '<p style="font-family:sans-serif; color:Blue; font-size: 15px;">Correlation between categorical and numerical👇'
    st.markdown(text_style_2,unsafe_allow_html=True)
    
    if data is not None: 
        all_columns_c=pd.DataFrame(df.loc[:,df.dtypes==np.object ]).columns                  
        high_cardi_columns = []
        normal_cardi_columns = []
        for i in all_columns_c:
            if (df[i].nunique() > df.shape[0] / 10):
                high_cardi_columns.append(i)
            else:
                    normal_cardi_columns.append(i)


        if len(normal_cardi_columns) == 0:
            st.write('There is no categorical columns with normal cardinality in the data.')
        else:
            
            model_type = st.radio('Select Visualize Type:', ('Scatter', 'Boxplot'), key = 'model_type')
            selected_cat_cols = functions.multiselect_container('Choose columns for Target Value:', normal_cardi_columns, 'Category')
                    
            
            target_cols= st.selectbox("Select Nummeric column:", n.columns, index = len(n.columns) - 1)    
            i = 0
            while (i < len(selected_cat_cols)):
                if model_type == 'Boxplot':
                    fig = px.box(df, y = target_cols, color = selected_cat_cols[i])
                else:
                    fig = px.scatter(df, color = selected_cat_cols[i], y = target_cols)

                st.plotly_chart(fig, use_container_width = True)
                i += 1

            if high_cardi_columns:
                if len(high_cardi_columns) == 1:
                    st.markdown('###### 🖇The following column has high cardinality, that is why its boxplot was not plotted:')
                else:
                    st.markdown('###### 🖇The following columns have high cardinality, that is why its boxplot was not plotted:')
                for i in high_cardi_columns:
                    st.write(i)
                
                select_columns=st.multiselect("Select high_cardi_columns",high_cardi_columns)
                st.write('<p style="font-size:120%">Do you want to plot anyway?</p>', unsafe_allow_html=True)    
                answer = st.selectbox("", ('No', 'Yes'))
                if answer == 'Yes':
                    for i in select_columns:
                        fig = px.box(df, y = target_cols, color = i)
                        st.plotly_chart(fig, use_container_width = True)
# multivariate
    st.markdown('##### 📌Multi-variate Analysis')
    if data is not None:
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), ax=ax)
        st.write(fig)


        





