import streamlit as st
import pandas as pd
import sklearn
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# Load and preprocess your data
df_cont = pd.read_csv('skindataall (1).csv')# Replace 'your_data.csv' with your actual data file
df_cont = df_cont[['Product', 'Product_id', 'Ingredients', 'Product_Url', 'Ing_Tfidf', 'Rating']]
df_cont.drop_duplicates(inplace=True)

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df_cont['Ingredients'])

# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

df_cont = df_cont.reset_index(drop=True)
titles = df_cont[['Product', 'Ing_Tfidf', 'Rating']]
indices = pd.Series(df_cont.index, index=df_cont['Product'])

# Recommendation function
def content_recommendations(product):
    idx = indices[product]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Exclude the product itself
    product_indices = [i[0] for i in sim_scores]
    return titles.iloc[product_indices]

# Streamlit UI with custom CSS for background image
st.markdown(
    """
    <style>
    body {
        background-image: url('skin.jpg');  # Set your background image URL
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Product Recommendation App")

product_name = st.text_input("Enter a product name:")

if st.button("Get Recommendations"):
    recommendations = content_recommendations(product_name)
    if not recommendations.empty:
        st.write("Top product recommendations based on content similarity:")
        st.dataframe(recommendations)
    else:
        st.write("No recommendations found for this product.")
