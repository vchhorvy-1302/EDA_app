
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

img=Image.open("my_app/logo.png")
st.set_page_config(page_title="Z1App/EDA/DataAnalysis", page_icon=img)


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

#End Theme


st.title("📈 Analysis")
# hide uploader file
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

activity = ["Quantitative","Qualitative","Check error"]
with st.sidebar:

    st.header("ANALYSIS WITH OPTION MENU")

    selected = option_menu(
    menu_title=None,  # required
    options=activity,  # required
    icons=None,  # optional
    menu_icon="menu-down",  # optional
    default_index=0,  # optional
    )

if data is not None:
    # Qualitative
    if selected=="Qualitative":
        st.markdown("#### Qualitative Analysis")
        st.markdown("##### 1️⃣ Categorical Data and Boolean Data")
        if data is not None:
            c=pd.DataFrame(df.loc[:,df.dtypes==np.object ])
            b=pd.DataFrame(df.select_dtypes(include=['bool']))
            f=pd.concat([c, b], axis=1)

            all_columns_c = pd.concat([c, b], axis=1).columns
            type_of_plot = st.radio("Select Type of Plot",["bar","pie"])
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

        with st.expander('📌Univariate Analysis'):  
        #Univariate
            st.markdown("##### 📌Univariate Analysis")
            if data is not None:
                n=pd.DataFrame(df.loc[:,(df.dtypes==np.int64) | (df.dtypes==np.float)])
                all_columns_n = n.columns
                type_of_plot = st.radio("Select Type of Plot",["box","hist"])
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
        with st.expander('📌Bivariate Analysis'):
        # Bivariate
            st.markdown('##### 📌Bivariate Analysis')
            text_style = '<p style="font-family:sans-serif; color:Blue; font-size: 15px;">➡️ Correlation between numerical and numcerial👇'
            st.markdown(text_style,unsafe_allow_html=True)
            if data is not None: 
                all_columns=n.columns                  
                selected_num_cols =st.selectbox('Choose column on X axis:', all_columns)  
                num_cols = st.selectbox("Choose column on Y axis:", n.columns, index = len(n.columns) - 1) 
                if st.button('Generate scatter plot'):   
                    fig = px.scatter(df, x= selected_num_cols, y = num_cols)
                    st.plotly_chart(fig)
            
            text_style_2 = '<p style="font-family:sans-serif; color:Blue; font-size: 15px;">➡️ Correlation between categorical and numerical👇'
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
                            st.markdown('###### ▶️ The following column has high cardinality, that is why its boxplot was not plotted:')
                        else:
                            st.markdown('###### ▶️ The following columns have high cardinality, that is why its boxplot was not plotted:')
                        for i in high_cardi_columns:
                            st.write(i)
                        
                        select_columns=st.multiselect("Select high_cardi_columns",high_cardi_columns)
                        st.write('<p style="font-size:100%">Do you want to plot anyway?</p>', unsafe_allow_html=True)    
                        answer = st.selectbox("", ('No', 'Yes'))
                        if answer == 'Yes':
                            for i in select_columns:
                                fig = px.box(df, y = target_cols, color = i)
                                st.plotly_chart(fig, use_container_width = True)
            st.markdown('###### ▶️ Scatter matrix')
            if data is not None:
                n_column=n.columns
                c_column=pd.DataFrame(df.loc[:,df.dtypes==np.object ]).columns
                target_column=st.selectbox("Select target categorical column", c_column)
                dimension=st.multiselect("Select matrix numerical columns",n_column)
                if st.button("Show graph"):
                    fig = px.scatter_matrix(df, dimensions=dimension, color=target_column)
                    st.plotly_chart(fig)
        with st.expander('📌Multi-variate Analysis'):
        # multivariate
            st.markdown('##### 📌Multi-variate Analysis')
            if data is not None:
                fig, ax = plt.subplots()
                sns.heatmap(df.corr(), ax=ax)
                st.write(fig)
    if selected=="Check error":
        column=df.columns
        st.markdown("### Check Land_area Calculation")
        c1, c2, c3 = st.columns(3)
        with c1:
            select_l=st.selectbox("Select land_length",column)
        with c2:
            select_w=st.selectbox("Select land_width",column)
        with c3:
            select_a=st.selectbox("Select land_area",column)
        df_l=df.loc[df[select_a].notnull()]
        # calculated land_area by multiply land_length and land_width
        df_l['true_land']= df_l[select_l]*df_l[select_w]
        #calculated the difference between land_area calculated and existing land_area columns
        df_size=df_l['true_land']-df_l[select_a]
        df_l["Correct_land"] = np.where((round(df_l['true_land'],1)==df_l[select_a]) | (df_size>-1) & (df_size<1), "True", "False")
        correct_area=df_l[df_l['Correct_land']=='True']['Correct_land'].dropna().count()
        not_correct_area=df_l[df_l['Correct_land']=='False']['Correct_land'].dropna().count()
        area=df_l[select_a].dropna().count()
        count_correct = pd.DataFrame({'correct_area':[correct_area/area*100],
                                'not_correct_area':[not_correct_area/area*100]
                                })
        correct_l = pd.DataFrame({'correct_area':[correct_area],
                                'not_correct_area':[not_correct_area]
                                })
        st.write(correct_l)

        fig = plt.figure(figsize=(10, 4))
        sns.set(rc={'figure.figsize':(11.7,8.27)})
        ax = sns.barplot(data=count_correct, palette = "Set2")
        ax.set_title(f"total number of coordinate points: {area}")
        for p in ax.patches:
            ax.text(p.get_x() + p.get_width()/2., p.get_height(), '{0:.2f}%'.format(p.get_height()), 
                fontsize=12, color='black', ha='center', va='bottom')
        st.pyplot(fig)
        if st.checkbox("Maximum land area of correct land area:"):
            st.write(df_l[df_l['Correct_land']=='True'][select_a].max())
        if st.checkbox("Minimum land area of correct land area:"):
            st.write(df_l[df_l['Correct_land']=='True'][select_a].min())


        st.markdown("### Check building_area Calculation")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            select_l_b=st.selectbox("Select building_length",column)
        with c2:
            select_w_b=st.selectbox("Select building_width",column)
        with c3:
            select_st=st.selectbox("Select stories building", column)
        with c4:
            select_ba=st.selectbox("Select buidling_area",column)
        df_b=df.loc[df[select_ba].notnull()]
        # calculated land_area by multiply land_length and land_width
        # calculated building_area by multiply building_length,building_width and stories
        df_b['true_building']= df_b[select_l_b]*df_b[select_w_b]*df_b[select_st]
        #calculated the difference between land_area calculated and existing land_area columns
        #df['df_size']=df_im['true_building']-df_im['building_area']
        df_b["Correct_building"] = np.where((round(df_b['true_building'],2)==df_b["building_area"]), "True", "False")
        correct_area=df_b[df_b['Correct_building']=='True']['Correct_building'].dropna().count()
        not_correct_area=df_b[df_b['Correct_building']=='False']['Correct_building'].dropna().count()
        area=df_b[select_ba].dropna().count()
        count_correct = pd.DataFrame({'correct_area':[correct_area/area*100],
                                'not_correct_area':[not_correct_area/area*100]
                                })

        correct_b = pd.DataFrame({'correct_area':[correct_area],
                                'not_correct_area':[not_correct_area]
                                })
        st.write(correct_b)
        fig = plt.figure(figsize=(10, 4))
        sns.set(rc={'figure.figsize':(11.7,8.27)})
        ax = sns.barplot(data=count_correct, palette = "Set2")
        ax.set_title(f"total number of coordinate points: {area}")
        for p in ax.patches:
            ax.text(p.get_x() + p.get_width()/2., p.get_height(), '{0:.2f}%'.format(p.get_height()), 
                fontsize=12, color='black', ha='center', va='bottom')
        st.pyplot(fig)

        if st.checkbox("Maximum building area of correct building area:"):
            st.write(df_b[df_b['Correct_building']=='True'][select_a].max())
        if st.checkbox("Minimum building area of correct building area:"):
            st.write(df_b[df_b['Correct_building']=='True'][select_a].min())

        st.markdown("### Check property structure")
        st.sidebar.markdown("##### Select Columns silder to fill structure:")
        
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
        
            modification_container =st.sidebar.container()

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

        c1, c2=st.columns([3,3])
       with c2:
            st.markdown("Check which type of property that have:")
            contain=st.selectbox("Select field",["Land_area","Building_area","Stories"])
        if contain=="Land_area":
            have_or_not=["contain land area","not contain land area"]
            if st.checkbox("contain land area"):
                df=df.loc[df[select_a].notnull()]
            else:
                st.checkbox("not contain land area")
                df=df.loc[df[select_a].isnull()]
        if contain=="Building_area":
            have_o_not=["contain building area","not contain building area"]
            if st.checkbox("contain building area"):
                df=df.loc[df[select_ba].notnull()]
            else :
                st.checkbox("not contain building area")
                df=df.loc[df[select_ba].isnull()]
        if contain=="Stories":
            have_o_not=["contain stories","not contain stories"]
            if st.checkbox("contain stories"):
                df=df.loc[df[select_st].notnull()]
            else :
                st.checkbox("not contain stories")
                df=df.loc[df[select_st].isnull()]
        
        
        with c1:
            st.markdown("Select property type columns to remove error")
            select_cols=st.multiselect("Select record_type, property_category,current_use",df.columns)
        st.dataframe(df.groupby(select_cols)['id'].count().reset_index())

        if st.checkbox("Zeros in Land_area"):
            st.write(df[df[select_a]==0][select_a].count())
        if st.checkbox("Zeros in Building area"):
            st.write(df[df[select_ba]==0][select_ba].count())
        if st.checkbox("Zeros in Stories"):
            st.write(df[df[select_st]==0][select_st].count())


            








        





