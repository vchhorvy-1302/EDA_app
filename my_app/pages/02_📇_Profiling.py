
import base64

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from st_aggrid import AgGrid
from pandas.api.types import (is_categorical_dtype, is_datetime64_any_dtype,
                              is_numeric_dtype, is_object_dtype)
from PIL import Image
from streamlit_option_menu import option_menu

from functions import functions

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
st.title("📇 Data Profiling ")


st.markdown('### 📁 Dataset')

with st.expander(""):
	file_format = st.radio('Select file format:', ('csv', 'excel'), key='file_format')
	data = st.file_uploader(label = '')

	use_defo = st.checkbox('Use example Dataset')
	if use_defo:
		data = 'my_app/sample_dataset.csv'


	if data:
		if file_format == 'csv' or use_defo:
			df = pd.read_csv(data)
		else:
			df = pd.read_excel(data)



st.subheader('Dataset Profile:')
with st.expander("Profiling"):  
    st.markdown("##### View Table 📑")
    if data is not None:
        AgGrid(df,height=500)
#         def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
#             modify = st.checkbox("Add filters")

#             if not modify:
#                 return df

#             df = df.copy()

#             # Try to convert datetimes into a standard format (datetime, no timezone)
#             for col in df.columns:
#                 if is_object_dtype(df[col]):
#                     try:
#                         df[col] = pd.to_datetime(df[col])
#                     except Exception:
#                         pass

#                 if is_datetime64_any_dtype(df[col]):
#                     df[col] = df[col].dt.tz_localize(None)

#             modification_container = st.container()

#             with modification_container:
#                 to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
#                 for column in to_filter_columns:
#                     left, right = st.columns((1, 20))
#                     left.write("↳")
#                     # Treat columns with < 45 unique values as categorical
#                     if is_categorical_dtype(df[column]) or df[column].nunique() < 20:
#                         user_cat_input = right.multiselect(
#                             f"Values for {column}",
#                             df[column].unique(),
#                             default=list(df[column].unique()),
#                         )
#                         df = df[df[column].isin(user_cat_input)]
#                     elif is_numeric_dtype(df[column]):
#                         _min = float(df[column].min())
#                         _max = float(df[column].max())
#                         step = (_max - _min) / 100
#                         user_num_input = right.slider(
#                             f"Values for {column}",
#                             _min,
#                             _max,
#                             (_min, _max),
#                             step=step,
#                         )
#                         df = df[df[column].between(*user_num_input)]
#                     elif is_datetime64_any_dtype(df[column]):
#                         user_date_input = right.date_input(
#                             f"Values for {column}",
#                             value=(
#                                 df[column].min(),
#                                 df[column].max(),
#                             ),
#                         )
#                         if len(user_date_input) == 2:
#                             user_date_input = tuple(map(pd.to_datetime, user_date_input))
#                             start_date, end_date = user_date_input
#                             df = df.loc[df[column].between(start_date, end_date)]
#                     else:
#                         user_text_input = right.text_input(
#                             f"Substring or regex in {column}",
#                         )
#                         if user_text_input:
#                             df = df[df[column].str.contains(user_text_input)]
#             return df
#         st._legacy_dataframe(filter_dataframe(df))
	

    # Select to show shape
        st.markdown("##### 📌Info:")
        @st.cache
        def info():
            n, m = df.shape
            st.write(f'<p style="font-size:120%,color:Blue;">Dataset contains {n} rows and {m} columns.</p>', unsafe_allow_html=True) 
            return 
        c1, c2, c3 = st.columns([3, 0.5, 0.5])
        dataTypeSeries = pd.DataFrame(df.dtypes).reset_index()
        dataTypeSeries.rename(columns={"index":"column",0:"Datatype"}, inplace=True)
        c1.dataframe(dataTypeSeries)


        st.markdown('##### 🔍Null Value Information:')
        if df.isnull().sum().sum() == 0:
            st.write('There is not any NA value in your dataset.')
        else:
            c1, c2, c3 = st.columns([4, 0.5, 0.5])
            functions.space(2)
            c1.dataframe(functions.df_isnull(df), width=1500)
        
        st.markdown('##### 📍Outlier Analysis')   
        c1, c2, c3 = st.columns([4, 0.5, 0.5])
        c1.dataframe(functions.number_of_outliers(df)) 
        st.markdown('##### Duplicate Rows')
        if st.checkbox('Duplicate Value'):
            st.write(df[df.duplicated(keep=False)])

activities = ["Categorical","Numerical","Date","Boolean"]	
with st.sidebar:

    st.header("PROFILING WITH OPTION MENU")

    selected = option_menu(
        menu_title=None,  # required
        options=activities,  # required
        icons=None,  # optional
        menu_icon="menu-down",  # optional
        default_index=0,  # optional
    )

# Categorical data
if selected=="Categorical":
    st.markdown("#### Categorical Data🔤")
    if data is not None:
        st.markdown("##### 1️⃣ View header")
        c=pd.DataFrame(df.loc[:,df.dtypes==np.object])
        st.write(c.head())
        
        if st.checkbox("Show Categorical Shape"):
            st.write(c.shape)

        st.markdown("##### 2️⃣ List Columns")
        st.write(c.columns.tolist())

        st.markdown("##### 3️⃣ Description")
        de=c.describe(include='all').fillna("").astype("str").T
        de['null_percentage']=c.isnull().sum()*100/len(c)
        st.write(de)


        if st.checkbox('Empty Cat_Columns'):
            e=pd.DataFrame(de[de["null_percentage"]==100]).T
            st.write(e.columns.tolist())

        if st.checkbox('Full Cat_Columns'):
            f=pd.DataFrame(de[de["null_percentage"]==0]).T
            st.write(f.columns.tolist())
        if st.checkbox('Unique Value'):
            all_columns_names = c.columns.tolist()
            selected_columns_names = st.selectbox("Select Categorical Columns To Plot",all_columns_names)
            q=pd.DataFrame(c[selected_columns_names].unique())
            st.write(q)
#End

# Numerical Data
if selected=="Numerical":
    st.markdown("#### Numerical Data 🔢")
    if data is not None:
        st.markdown("##### 1️⃣ View header")
        n=pd.DataFrame(df.loc[:,(df.dtypes==np.int64) | (df.dtypes==np.float)])
        st.write(n.head())

        if st.checkbox("Show Numerical Shape"):
            st.write(n.shape)

        st.markdown("##### 2️⃣ List Columns")
        st.write(n.columns.tolist())

        st.markdown("##### 3️⃣ Description")
        de=n.describe(include='all').T
		
        de["Lower_fence"]=de['75%']-(1.5 * (de['75%']-de['25%'])) 
        de["Upper_fence"]=(de['75%']+(1.5 * (de['75%']-de['25%'])))
        de['null_percentage']=n.isnull().sum()*100/len(n)
        st.write(de)

        if st.checkbox('Empty Num_Columns'):
            e=pd.DataFrame(de[de["null_percentage"]==100]).T
            st.write(e.columns.tolist())

        if st.checkbox('Full Num_Columns'):
            f=pd.DataFrame(de[de["null_percentage"]==0]).T
            st.write(f.columns.tolist())

# End

# Date Data

if selected=="Date":
    st.markdown("#### Date Data 📆")
    if data is not None:
        st.markdown("##### 1️⃣ View header")
        d=df.select_dtypes(include=['datetime64'])
        d_columns=d.columns
        if len(d_columns) == 0:
            st.write('There is no date type columns in the data.')
        else:
            st.write(d.head())

        if st.checkbox("Show Date Shape"):
            st.write(d.shape)

            st.markdown("##### 2️⃣ List Columns")
            st.write(d.columns.tolist())

            st.markdown("##### 3️⃣ Description")
            
            de=d.describe(include='all').T
            de['null_percentage']=d.isnull().sum()*100/len(d)
            st.write(de)
        
        if d is not None:
            de=d.describe(include='all').T
            st.checkbox('Empty Columns')
            e=pd.DataFrame(de[de["null_percentage"]==100]).T
            st.write(e.columns.tolist())
        else:
            st.markdown('No date columns')
        if d is not None:
            de=d.describe(include='all').T
            st.checkbox('Full Columns')
            f=pd.DataFrame(de[de["null_percentage"]==0]).T
            st.write(f.columns.tolist())
        else:
            st.markdown('No date columns')


# Boolean Data

if selected=="Boolean":
    st.markdown("#### Boolean Data ☑️ ✖️")
    if data is not None:
        st.markdown("##### 1️⃣ View header")
        b=pd.DataFrame(df.select_dtypes(include=['bool']))
        b_columns=pd.DataFrame(df.select_dtypes(include=['bool'])).columns
        if len(b_columns) == 0:
            st.write('There is no boolean type columns in the data.')
        else:
            st.write(b.head())
        if st.checkbox("Show Boolean Shape"):
            st.write(b.shape)

            st.markdown("##### 2️⃣ List Columns")
            st.write(b.columns.tolist())

        st.markdown("##### 3️⃣ Description")
            
        de=b.describe(include='all').T
        de['null_percentage']=b.isnull().sum()*100/len(b)
        st.write(de)
    
        if st.checkbox('Empty Columns'):
            e=pd.DataFrame(de[de["null_percentage"]==100]).T
            st.write(e.columns.tolist())

        if st.checkbox('Full Columns'):
            f=pd.DataFrame(de[de["null_percentage"]==0]).T
            st.write(f.columns.tolist())
        if st.checkbox('Unique Value'):
            all_columns_names = b.columns.tolist()
            selected_columns_names = st.selectbox("Select Categorical Columns To Plot",all_columns_names)
            st.write(b[selected_columns_names].unique())



    


    
