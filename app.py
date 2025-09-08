import streamlit as st
import pandas as pd
import pickle
brain = pickle.load(open('Shoppe.pkl', 'rb'))
item_similarity_df = pd.read_pickle("item_similarity.pkl")

# ---------- Define cluster -> segment mapping ----------
cluster_labels = {
    0: "Occasional",
    1: "High-Value",
    2: "Regular",
    3: "At Risk"
}

# ---------- Function -> Segment Prediction ----------
def predict_segment_from_df(recency, frequency, monetary):
    input_data = pd.DataFrame([[recency, frequency, monetary]],
                              columns=["Recency", "Frequency", "Monetary"])
    cluster = brain.predict(input_data)[0]
    segment = cluster_labels.get(cluster, "Unknown")
    return cluster, segment

# ---------- Function -> Product Recommendation ----------
def get_similar_products(product, top_n=5):
    # Check if product exists in similarity matrix
    if product not in item_similarity_df.columns:
        return None
    # Get similarity scores
    similar_scores = item_similarity_df[product].sort_values(ascending=False)
    # Exclude the product itself
    similar_scores = similar_scores.drop(product)
    return similar_scores.head(top_n)

st.sidebar.title("🛍️📊 Shopper Spectrum")
st.sidebar.divider()
menu = st.sidebar.radio("Navigate",("Clustering", "Recommendation"))
if menu == "Recommendation":
  st.title("🛒 Product Recommendation System")
  st.write("Item-based Collaborative Filtering using purchase history")

  # Search box
  search_product = st.text_input("Enter a product name:")
  button = st.button("Recommend")
  if button:
    if search_product:
        results = get_similar_products(search_product, top_n=5)
        if results is None:
          st.warning("⚠️ No matching product found. Try a different name.")
        else:
            st.subheader("Top 5 Similar Products:")
            for desc in results.index:
                st.write(f"- {desc}")
elif menu == "Clustering":
  st.title("👥📊 Customer Segmentation")
  st.write("Enter Recency, Frequency and Monetary values to predict the customer's cluster and segment")
  # Input fields
  recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=1000, value=0)
  frequency = st.number_input("Frequency (number of purchases)", min_value=0, max_value=500, value=0)
  monetary = st.number_input("Monetary (total spend)", min_value=0.0, max_value=100000.0, value=0.0, step=10.0)
  if st.button("Predict Segment"):
    cluster, segment = predict_segment_from_df(recency, frequency, monetary)
    st.success(f"🧩 Predicted Cluster: **{cluster}**")
    st.success(f"🎯 Predicted Segment: **{segment}**")
