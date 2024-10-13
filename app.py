import streamlit as st
import pandas as pd
import numpy as np
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

# Load the Naive Bayes model
@st.cache_resource
def load_model():
    with open('nb_model.pkl', 'rb') as file:
        return pickle.load(file)

# Load the TF-IDF vectorizer (if used)
@st.cache_resource
def load_vectorizer():
    with open('tfidf_vectorizer.pkl', 'rb') as file:
        return pickle.load(file)

# Preprocess the input email text using CountVectorizer (Bag of Words)
def preprocess_text(text, vectorizer):
    return vectorizer.transform([text])  # Transform the input email text into Bag of Words representation

# Generate word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Set2').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

# Predict the label of the input email using Naive Bayes
def predict_email(model, vectorizer, email_text):
    processed_email = preprocess_text(email_text, vectorizer)
    prediction = model.predict(processed_email)
    label = prediction[0]
    label_mapping = {0: "Ham", 1: "Spam"}
    prediction_probs = model.predict_proba(processed_email)  # Get probabilities for each class
    return label_mapping[label], prediction_probs

# Main app
def main():
    # Set up the app title
    st.title("Spam Email Detection App 📧")
    st.write("This app predicts whether an email is **Ham** or **Spam**.")

    # Load the model and vectorizer
    model = load_model()
    vectorizer = load_vectorizer()

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Go to", ["Home", "Predict"])

    if options == "Home":
        st.subheader("About the App")
        st.write("This app uses a trained Naive Bayes model to classify emails into one of two categories: **Ham** or **Spam**. It also provides visual insights through colorful plots and a word cloud.")
        
        # Load dataset with error handling
        st.subheader("Dataset Overview")
        # Path to the zip file and the CSV file inside the zip archive
        zip_file_path = 'Spam_email.zip'
        csv_filename = 'Spam_email.csv'  # The name of the CSV file inside the zip

        # Open the zip file and read the CSV file
        with zipfile.ZipFile(zip_file_path, 'r') as z:
            with z.open(csv_filename) as f:
                df = pd.read_csv(f, on_bad_lines='skip')

        # Now, 'df' will contain the loaded CSV data
        df['Label'] = df['Label'].replace({'Spam': 1, 'Ham': 0})
        st.dataframe(df.head())

        # Visualizing label distribution
        st.subheader("Label Distribution")
        label_counts = df['Label'].value_counts().sort_index()
        category_names = ['Ham (0)', 'Spam (1)']
        fig, ax = plt.subplots()
        sns.barplot(x=category_names, y=label_counts, palette='Set2', ax=ax)
        ax.set_title('Email Type Distribution')
        ax.set_xlabel('Email Type')
        ax.set_ylabel('Count')
        st.pyplot(fig)

        # Visualizing email length distribution
        st.subheader("Email Length Distribution")
        email_lengths = df['Cleaned_Email'].apply(len)
        fig, ax = plt.subplots()
        sns.histplot(email_lengths, kde=True, ax=ax, color='skyblue')
        ax.set_title('Distribution of Email Lengths')
        ax.set_xlabel('Email Length')
        st.pyplot(fig)

    elif options == "Predict":
        st.subheader("Predict Email Type")

        # Input for the email text
        email_text = st.text_area("Enter the email text below:")
        
        if st.button("Predict"):
            if email_text.strip() == "":
                st.warning("Please enter some email text!")
            else:
                # Predict the email type
                label, prediction_probs = predict_email(model, vectorizer, email_text)
                st.success(f"The email is classified as: **{label}**")
                
                # Show probabilities for each class
                st.write("Prediction Probabilities:")
                st.write(f"**Ham**: {prediction_probs[0][0]*100:.2f}%")
                st.write(f"**Spam**: {prediction_probs[0][1]*100:.2f}%")
                
                # Generate word cloud
                st.subheader("Word Cloud of the Email")
                generate_wordcloud(email_text)

if __name__ == "__main__":
    main()
