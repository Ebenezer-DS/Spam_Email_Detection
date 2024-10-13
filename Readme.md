**# Spam Email Detection App**

This repository contains the code for a Spam Email Detection App, built using Streamlit. The app predicts whether an email is classified as either Ham or Spam, based on the email content. The app leverages machine learning models using Naive Bayes, combined with TF-IDF text processing techniques.

**# Features**
__**Classification**: The app predicts whether an email is:__
    __**Ham**: Legitimate, non-spam emails (including both regular and promotional emails).__
    __**Spam**: Unsolicited or irrelevant emails__
__**Word Cloud**: Generates a word cloud visualization of the entered email content.__
__**Data Insights**: Provides insights into the dataset with label distribution and email length visualizations.__

**# Dataset**
The dataset used for training and evaluation includes two types of emails: Spam and Ham. It is preprocessed, and each email is transformed using a TF-IDF vectorizer or Bag of Words method.

**# Folder Structure**
- Spam_Email_Detection/
  - app.py               # Streamlit web app
  - Spam_email.csv        # Dataset
  - nb_model.pkl          # Pre-trained Naive Bayes model (optional SVM model can be used)
  - tfidf_vectorizer.pkl  # TF-IDF Vectorizer
  - README.md             # Project README file

**# Installation**
1. Clone the repository:
git clone https://github.com/Ebenezer-DS/Spam_Email_Detection.git

2. Navigate to the project directory:
cd Spam_Email_Detection

3. Install the required packages:
pip install -r requirements.txt

**# Usage**

**# Running the Streamlit App**
To launch the Streamlit app, run:
streamlit run app.py

This will start the app on your local machine. You can access it by opening your web browser and going to http://localhost:8501.

**# Main App (app.py)**
The app.py file contains the logic for the Spam Email Detection app. Here’s what the app does:

1. **Loading the Model and Vectorizer:**

The app loads a pre-trained Naive Bayes(e.g., nb_model.pkl) and the TF-IDF vectorizer (tfidf_vectorizer.pkl).

2. **Preprocessing Input:**

When the user inputs email text, the app preprocesses it using the loaded TF-IDF vectorizer.

3. **Prediction:**

The preprocessed text is passed to the model to predict whether the email is Ham or Spam.

4. **Visualization:**

The app also provides visualizations such as word clouds, label distribution, and email length distribution for a more interactive experience.

**# Example Emails**
**Here are examples of the types of emails classified by the app:**

**# Ham Example:**
Subject: Meeting Confirmation
Hey, just confirming our meeting at 2 PM tomorrow at the office. Let me know if the time works for you.
Best regards,
John

**# Spam Example:**
Subject: Congratulations! You've Won a Free Cruise!
Dear Sir/Madam,
You’ve been selected as the lucky winner of a FREE CRUISE to the Bahamas! Click the link below to claim your prize.
Hurry, this offer won’t last long!
Claim Now
Best regards,
The Free Cruise Team

By merging Easy Ham and Hard Ham into a single Ham category, the app focuses on distinguishing between legitimate emails (Ham) and unsolicited or irrelevant emails (Spam).






