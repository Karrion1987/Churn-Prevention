import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import random

# Load dataset
file_path = "data/dataset_transformado.parquet"
data = pd.read_parquet(file_path)

# Drop 'Purchase Date' column
data.drop(columns=['Purchase Date'], inplace=True)

# Assuming 'Churn' is the target column
X = data.drop(columns=['Churn'])
y = data['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier for churn prediction
@st.cache(allow_output_mutation=True)
def train_classifier():
    churn_clf = RandomForestClassifier()
    churn_clf.fit(X_train, y_train)
    return churn_clf

churn_clf = train_classifier()

# Product information dictionary
product_info = {
    "Electronics": ["Smartphone", "Laptop", "Headphones", "Tablet", "Smartwatch"],
    "Home": ["Sofa", "Bed", "Dining table", "Television", "Vacuum cleaner"],
    "Clothing": ["T-shirt", "Jeans", "Dress", "Sweater", "Jacket"],
    "Books": ["Novel", "Self-help book", "Textbook", "Biography", "Cookbook"]
}

# Function to recommend categories based on churn prediction and product information
@st.cache
def recommend_category(churn, product_info, abandoned_category):
    if churn == 1:  # If churn is predicted
        # Recommend similar category
        similar_category = random.choice(list(product_info.keys()))
        recommendation = f"We recommend exploring more products from the {similar_category} category."
        recommended_products = product_info.get(similar_category, [])
    else:
        # Recommend dissimilar category
        dissimilar_category = random.choice(list(product_info.keys()))
        recommendation = f"We recommend exploring products from the {dissimilar_category} category, as they are different from what you've previously viewed."
        recommended_products = product_info.get(dissimilar_category, [])
    return recommendation, recommended_products

# Streamlit UI
st.title("Category Recommendation System")
st.image("data/Red Modern Market Logo (1).png", use_column_width=True)

# Input field for churn prediction
churn_input = st.number_input("Enter churn prediction (1 for churn, 0 for no churn)", min_value=0, max_value=1, value=0, step=1)

# Input field for abandoned category
abandoned_category = st.selectbox("Select the abandoned category", list(product_info.keys()))

# Button to trigger recommendation
if st.button("Get Recommendation"):
    try:
        # Get recommendation based on churn prediction
        recommendation, recommended_products = recommend_category(churn_input, product_info, abandoned_category)
        st.write("Recommendation:", recommendation)
        st.write("Some products from the recommended category:")
        for product in recommended_products:
            st.write("-", product)
    except Exception as e:
        st.error("An error occurred: {}".format(str(e)))
