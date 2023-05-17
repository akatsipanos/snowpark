# Snowpark
from snowflake.snowpark.session import Session
import snowflake.snowpark.functions as F
# Misc
import pandas as pd
import json
import altair as alt
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 
st.set_page_config(
     page_title="Credit Card Approval Prediciton",
     page_icon="💳",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://developers.snowflake.com',
         'About': "This is an *extremely* cool app powered by Snowpark for Python, Snowflake Data Marketplace and Streamlit"
     }
)

def create_session():
    if "snowpark_session" not in st.session_state:
        connection_parameters = json.load(open('connection.json'))
        session = Session.builder.configs(connection_parameters).create()
        st.session_state['snowpark_session'] = session
    else:
        session = st.session_state['snowpark_session']
    return session

@st.cache_data()
def load_data():
    # Load in Credit Data
    credit_df = session.table('CREDIT_RISK_PREPARED_BALANCED_TRAIN_SCORED_XGB')
    return credit_df.to_pandas()


st.markdown("<h1 style='margin-top:-80px;'>Credit Card Approval Prediction</h1>", unsafe_allow_html=True)
session = create_session()
credit_df = load_data()

def main_page():
    st.subheader('Feature Importance')
    model_file = session.file.get('@ml_models/xgb_model.sav', 'tmp')
    model = joblib.load(f'tmp/{model_file[0].file}')

    feature_cols = credit_df.drop(['TARGET', 'PREDICTION'],axis=1).columns
    feature_importance = pd.DataFrame(data = model.feature_importances_, 
                                      index = feature_cols, 
                                      columns=['Feature_importance'])
    feature_importance = feature_importance.sort_values('Feature_importance', ascending=False)

    selected_features = st.multiselect('',feature_cols)
    st.markdown('___')

    with st.container():
        st.write('Select features to display their feature importance on the bar chart. The top 10 features are shown if none are selected')
        feature_list = feature_importance.index[:10] if len(selected_features) == 0 else selected_features

        feature_importance_filtered = feature_importance[feature_importance.index.isin(feature_list)]

        fig,ax = plt.subplots()
        sns.barplot(data = feature_importance_filtered,x = 'Feature_importance', y=feature_importance_filtered.index, ax=ax)
        st.pyplot(fig, use_container_width=True)
    
def page_two():
    st.subheader('Exploratory Data Analysis')
    st.write('Example of EDA that could be shown')

    application_record_sdf = session.table('APPLICATION_RECORD')
    var_analysis = application_record_sdf.group_by('NAME_EDUCATION_TYPE').agg(F.count('NAME_EDUCATION_TYPE').as_('COUNT'))
    var_analysis = var_analysis.sort('COUNT').to_pandas()
    fig, ax = plt.subplots()
    var_analysis.plot.barh(x='NAME_EDUCATION_TYPE', ax = ax, xlabel = 'Education Type')
    ax.set_title('Education Type')
    ax.legend(loc = 'lower right')
    st.pyplot(fig, use_container_width=True)



page_names_to_funcs = {
    "Feature Importance": main_page,
    "EDA": page_two}

selected_page = st.sidebar.selectbox("Select", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()