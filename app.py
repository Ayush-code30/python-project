import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set up Gemini
genai.configure(api_key="YOUR_GOOGLE_API_KEY")  # Replace with your actual API key

def get_gemini_response(prompt_intro, review_text, _placeholder=""):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([prompt_intro, review_text, _placeholder])
    return response.text

# Streamlit page setup
st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊")
st.title("😊😠 Product Review Sentiment Analyzer")
st.write("Classify reviews as Positive, Neutral, or Negative using ML or Gemini")

# Sample dataset
@st.cache_data
def load_data():
    data = pd.DataFrame({
        "review": [
            "I love this product! Amazing quality.",
            "It's okay, but could be better.",
            "Terrible experience. Would not buy again.",
            "Great value for money.",
            "Not what I expected.",
            "Perfect! Exactly as described.",
            "Waste of money.",
            "Average product, nothing special.",
            "Highly recommend!",
            "Disappointed with the purchase."
        ],
        "sentiment": ["positive", "neutral", "negative", "positive", "neutral",
                      "positive", "negative", "neutral", "positive", "negative"]
    })
    return data

data = load_data()

# Show sample data
if st.checkbox("Show sample reviews"):
    st.write(data)

# Preprocess and train ML model
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(data["review"])
y = data["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Show accuracy
st.metric("Model Accuracy", f"{accuracy:.0%}")

# Choose prediction method
method = st.radio("Choose prediction method:", ("Naive Bayes (ML)", "Gemini AI"))

# Input for prediction
st.subheader("🔍 Try It Yourself")
user_review = st.text_area("Enter a product review:", "This product is great!")

if st.button("Predict Sentiment"):
    if method == "Naive Bayes (ML)":
        review_vec = vectorizer.transform([user_review])
        prediction = model.predict(review_vec)[0]
        proba = model.predict_proba(review_vec).max()

        if prediction == "positive":
            st.success(f"😊 Positive ({proba:.0%} confidence)")
        elif prediction == "neutral":
            st.info(f"😐 Neutral ({proba:.0%} confidence)")
        else:
            st.error(f"😠 Negative ({proba:.0%} confidence)")

    elif method == "Gemini AI":
        with st.spinner("Contacting Gemini..."):
            prompt = (
                "Classify the sentiment of this product review as Positive, Neutral, or Negative. "
                "Just return one word (Positive, Neutral, or Negative), and if possible, also give a short confidence estimate.\n"
                "Review:"
            )
            gemini_output = get_gemini_response(prompt, user_review)

        # Basic formatting for response
        st.subheader("Gemini's Response:")
        if "positive" in gemini_output.lower():
            st.success(f"😊 {gemini_output}")
        elif "neutral" in gemini_output.lower():
            st.info(f"😐 {gemini_output}")
        elif "negative" in gemini_output.lower():
            st.error(f"😠 {gemini_output}")
        else:
            st.warning(f"🤖 Unclear response: {gemini_output}")

# Footer
st.markdown("---")
st.caption("Internship Project | Sentiment Analysis with Naive Bayes + Gemini AI")
