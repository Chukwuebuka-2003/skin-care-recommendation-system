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

# Options for Skin_Tone, Skin_Type, Eye_Color, and Hair_Color
skin_tone_options = ["No data", "Light", "Fair", "Medium", "Olive", "Tan", "Porcelain", "Deep", "Dark", "Ebony"]
skin_type_options = ["Combination", "No data", "Dry", "Oily", "Normal"]
eye_color_options = ["Brown", "No data", "Blue", "Hazel", "Green", "Gray"]
hair_color_options = ["Brunette", "No data", "Blonde", "Black", "Auburn", "Red", "Gray"]

skintone = st.sidebar.selectbox("Skin Tone:", skin_tone_options)
skintype = st.sidebar.selectbox("Skin Type:", skin_type_options)
eyecolor = st.sidebar.selectbox("Eye Color:", eye_color_options)
haircolor = st.sidebar.selectbox("Hair Color:", hair_color_options)

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
