import pandas as pd
import streamlit as st
import pickle
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('skindataall (1).csv')

# Load the saved SVD model
with open('svd_model.pkl', 'rb') as model_file:
    svd = pickle.load(model_file)

# Create Streamlit UI
st.title("Product Recommendation App")

# Input Fields
st.sidebar.header("User Features")
skintone = st.sidebar.text_input("Skin Tone:")
skintype = st.sidebar.text_input("Skin Type:")
eyecolor = st.sidebar.text_input("Eye Color:")
haircolor = st.sidebar.text_input("Hair Color")

if st.sidebar.button("Get Recommendations"):
    # Function to make predictions
    def recommend_products_by_user_features(skintone, skintype, eyecolor, haircolor, percentile=0.85):
        ddf = df[(df['Skin_Tone'] == skintone) & (df['Hair_Color'] == haircolor) & (df['Skin_Type'] == skintype) & (df['Eye_Color'] == eyecolor)]

        # Perform SVD collaborative filtering on the filtered data
        recommendations = []

        for _, row in ddf.iterrows():
            prediction = svd.predict(row['User_id'], row['Product_id'])
            recommendations.append({'Rating_Stars': prediction.est, 'Product_Url': row['Product_Url'], 'Product': row['Product']})

        recommendations_df = pd.DataFrame(recommendations)
        recommendations_df = recommendations_df.sort_values('Rating_Stars', ascending=False).head(10)

        st.subheader('Based on your features, these are the top products for you:')
        st.table(recommendations_df)

    recommend_products_by_user_features(skintone, skintype, eyecolor, haircolor)
