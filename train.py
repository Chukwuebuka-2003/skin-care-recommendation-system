import pandas as pd
import streamlit as st
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('skindataall (1).csv')

# Create Streamlit UI
st.title("Product Recommendation App")

# Input Fields
st.sidebar.header("User Features")
skintone = st.sidebar.text_input("Skin Tone:")
skintype = st.sidebar.text_input("Skin Type:")
eyecolor = st.sidebar.text_input("Eye Color:")
haircolor = st.sidebar.text_input("Hair Color")

if st.sidebar.button("Get Recommendations"):
    # Filter recommendations based on user features
    ddf = df[(df['Skin_Tone'] == skintone) & (df['Hair_Color'] == haircolor) & (df['Skin_Type'] == skintype) & (df['Eye_Color'] == eyecolor)]
    recommendations = ddf[ddf['Rating_Stars'].notnull()][['Rating_Stars', 'Product_Url', 'Product']]
    recommendations = recommendations.sort_values('Rating_Stars', ascending=False).head(10)

    st.subheader('Based on your features, these are the top products for you:')
    st.table(recommendations)

# Define the 'data' variable for SVD-based recommendation
reader = Reader(line_format='user item rating', sep=',')
data = Dataset.load_from_df(df[['User_id', 'Product_id', 'Rating']], reader=reader)

# Perform SVD-based recommendation on user interaction
st.sidebar.header("SVD-Based Recommendation")
trainset, testset = train_test_split(data, test_size=.2)

svd = SVD()
svd.fit(trainset)

predictions = svd.test(testset)
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

st.write(f"RMSE: {rmse}")
st.write(f"MAE: {mae}")

# Recommendations for a specific user
st.sidebar.header("SVD-Based User Recommendations")
user_skintone = st.sidebar.text_input("User's Skin Tone:")
user_skintype = st.sidebar.text_input("User's Skin Type:")
user_eyecolor = st.sidebar.text_input("User's Eye Color:")
user_haircolor = st.sidebar.text_input("User's Hair Color")

if st.sidebar.button("Get User Recommendations"):
    # Function definition for user-specific recommendations
    def recommend_products_by_user_features(skintone, skintype, eyecolor, haircolor, percentile=0.85):
        ddf = df[(df['Skin_Tone'] == skintone) & (df['Hair_Color'] == haircolor) & (df['Skin_Type'] == skintype) & (df['Eye_Color'] == eyecolor)]

        recommendations = ddf[ddf['Rating_Stars'].notnull()][['Rating_Stars', 'Product_Url', 'Product']]
        recommendations = recommendations.sort_values('Rating_Stars', ascending=False).head(10)

        return recommendations

    user_recommendations = recommend_products_by_user_features(user_skintone, user_skintype, user_eyecolor, user_haircolor)
    st.subheader('User-specific product recommendations:')
    st.table(user_recommendations)
