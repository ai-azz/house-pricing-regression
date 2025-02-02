import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# page configuration
st.set_page_config(
    page_title="House Price EDA",
    page_icon="🏠",
    layout="wide"
)

# setting theme and lang
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
    
if 'language' not in st.session_state:
    st.session_state.language = 'id'

# theme func
# theme func
# theme func
def apply_theme():
    if st.session_state.theme == 'dark':
        plt.style.use('dark_background')  
        sns.set_style("darkgrid")         
    else:
        plt.style.use('ggplot')           
        sns.set_style("whitegrid")        

translations = {
    'id': {
        'title': "Analisis Harga Rumah 🏘️",
        'description': "Dashboard ini menampilkan eksplorasi data dari dataset harga rumah (House Prices - Advanced Regression Techniques).",
        'raw_data': "Lihat Data Mentah",
        'price_dist': "Distribusi Harga Rumah",
        'corr_analysis': "Analisis Korelasi",
        'feature_rel': "Hubungan Fitur dengan Harga",
        'num_feature': "Pilih Fitur Numerik",
        'cat_analysis': "Analisis Fitur Kategorikal",
        'cat_feature': "Pilih Fitur Kategorikal",
        'insight_title': "📊 Insight Analisis",
        'insight_1': "Hubungan positif kuat antara luas ruang tinggal (GrLivArea) dan harga jual",
        'insight_2': "Korelasi moderat antara luas garasi (GarageArea) dan harga jual",
        'insight_3': "Outliers mempengaruhi analisis - pertimbangkan transformasi logaritmik",
        'insight_4': "Ukuran properti sebagai faktor utama penentu harga",
        'insight_5': "Perlu analisis khusus untuk outlier ekstrim",
        'warning': "⚠️ Tidak ada fitur kategorikal dalam dataset yang telah dibersihkan"
    },
    'en': {
        'title': "House Price Analysis 🏘️",
        'description': "This dashboard shows exploratory data analysis from house prices dataset (House Prices - Advanced Regression Techniques).",
        'raw_data': "View Raw Data",
        'price_dist': "House Price Distribution",
        'corr_analysis': "Correlation Analysis",
        'feature_rel': "Feature-Price Relationship",
        'num_feature': "Select Numeric Feature",
        'cat_analysis': "Categorical Feature Analysis",
        'cat_feature': "Select Categorical Feature",
        'insight_title': "📊 Analysis Insights",
        'insight_1': "Strong positive correlation between living area (GrLivArea) and sale price",
        'insight_2': "Moderate correlation between garage area (GarageArea) and sale price",
        'insight_3': "Outliers affect analysis - consider logarithmic transformation",
        'insight_4': "Property size as main price determinant",
        'insight_5': "Special analysis needed for extreme outliers",
        'warning': "⚠️ There are no categorical features in the cleaned dataset"
    }
}

@st.cache_data
def load_data():
    return pd.read_csv('cleaned_data.csv')

df = load_data()

with st.sidebar:
    st.header("⚙️ Pengaturan")
    
    theme = st.radio(
        "Tema",
        ['Terang', 'Gelap'],
        index=0 if st.session_state.theme == 'light' else 1,
        key='theme_toggle',
        format_func=lambda x: '🌞 Light' if x == 'Terang' else '🌚 Dark'
    )
    st.session_state.theme = 'light' if theme == 'Terang' else 'dark'
    
    language = st.radio(
        "Bahasa",
        ['Indonesia', 'English'],
        index=0 if st.session_state.language == 'id' else 1,
        key='lang_toggle'
    )
    st.session_state.language = 'id' if language == 'Indonesia' else 'en'

apply_theme()

lang = st.session_state.language
st.title(translations[lang]['title'])
st.markdown(translations[lang]['description'])

with st.expander(translations[lang]['raw_data']):
    st.dataframe(df)

# section 1: price distribution
st.header(translations[lang]['price_dist'])
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.histplot(df['SalePrice'], kde=True, ax=ax)
    ax.set_title(translations[lang]['price_dist'])
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.boxplot(x=df['SalePrice'])
    ax.set_title(translations[lang]['price_dist'])
    st.pyplot(fig)

# insight section
with st.expander(translations[lang]['insight_title']):
    st.markdown(f"""
    - **{translations[lang]['insight_1']}**
    - **{translations[lang]['insight_3']}**
    - **{translations[lang]['insight_5']}**
    """)

# section 2: correlation
st.header(translations[lang]['corr_analysis'])
corr_matrix = df.corr(numeric_only=True)

fig = px.imshow(
    corr_matrix,
    labels=dict(color=translations[lang]['corr_analysis']),
    x=corr_matrix.columns,
    y=corr_matrix.columns
)
st.plotly_chart(fig, use_container_width=True)

with st.expander(translations[lang]['insight_title']):
    st.markdown(f"""
    - **{translations[lang]['insight_2']}**
    - **{translations[lang]['insight_4']}**
    """)

# section 3: price feature relations
st.header(translations[lang]['feature_rel'])
selected_feature = st.selectbox(
        translations[st.session_state.language]['num_feature'],
        df.select_dtypes(include=['int64', 'float64']).columns
    )

col1, col2 = st.columns(2)
with col1:
    fig = px.scatter(
        df,
        x=selected_feature,
        y='SalePrice',
        trendline="ols"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    top_corr_features = corr_matrix['SalePrice'].abs().sort_values(ascending=False).index[1:6]
    fig = px.bar(
        x=top_corr_features,
        y=corr_matrix['SalePrice'][top_corr_features].abs(),
        labels={'x': translations[lang]['num_feature'], 
                'y': translations[lang]['corr_analysis']}
    )
    st.plotly_chart(fig, use_container_width=True)

# section 4: categorical analysis
st.header(translations[lang]['cat_analysis'])
categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    categorical_feature = st.selectbox(
        translations[lang]['cat_feature'],
        categorical_cols
    )
    
    fig = px.box(
        df,
        x=categorical_feature,
        y='SalePrice',
        color=categorical_feature
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning(translations[lang]['warning'])

# custom css
theme_css = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: %s;
    color: %s;
}
</style>
"""

if st.session_state.theme == 'dark':
    bg_color = "#0E1117"
    text_color = "#FFFFFF"
else:
    bg_color = "#FFFFFF"
    text_color = "#000000"

st.markdown(theme_css % (bg_color, text_color), unsafe_allow_html=True)