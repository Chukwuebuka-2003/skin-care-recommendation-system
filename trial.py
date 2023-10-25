import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# Load and preprocess your data
df_cont = pd.read_csv('skindataall (1).csv')  # Replace 'your_data.csv' with your actual data file
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

# Load the SVD model
with open('svd_model.pkl', 'rb') as model_file:
    svd = pickle.load(model_file)

# Recommendation function using SVD
def svd_recommendations(user_id):
    # This function should return product recommendations for the given user_id using SVD
    # You can use svd.predict() to get ratings for products and recommend the top-rated ones.
    pass

# Streamlit UI
st.title("Product Recommendation App")

product_name = st.text_input("Enter a product name:")

if st.button("Get Content-Based Recommendations"):
    content_recommendations = content_recommendations(product_name)
    if not content_recommendations.empty:
        st.write("Top product recommendations based on content similarity:")
        st.dataframe(content_recommendations)
    else:
        st.write("No recommendations found for this product.")

user_id = st.text_input("Enter your user ID:")  # You need a way to input the user ID
if st.button("Get Collaborative Filtering Recommendations"):
    svd_recommendations = svd_recommendations(user_id)
    if svd_recommendations:
        st.write("Top product recommendations based on collaborative filtering:")
        st.dataframe(svd_recommendations)
    else:
        st.write("No recommendations found for this user.")
