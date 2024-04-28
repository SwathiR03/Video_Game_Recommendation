import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('vgsales.csv')

# Remove missing values
df.dropna(inplace=True)

# Remove special characters from game names
df['Name'] = df['Name'].str.replace('[^\w\s]', '', regex=True)

# Create a Streamlit app
st.title("Game Recommendation System")

# Take input from user
preferred_genre = st.text_input("Enter your preferred genre:")
preferred_platform = st.text_input("Enter your preferred platform:")

# Filter the dataset based on user input
df_filtered = df[(df['Genre'] == preferred_genre) & (df['Platform'] == preferred_platform)]

# If no matching games are found
if df_filtered.empty:
    st.write("Sorry, no matching games found for your preferences.")
else:
    # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer()

    # Apply vectorization on the game names
    tfidf_matrix = vectorizer.fit_transform(df_filtered['Name'])

    # Calculate cosine similarity between the game names
    similarity_scores = cosine_similarity(tfidf_matrix)

    # Get index of the game entered by the user
    index = df_filtered.index[0]

    # Get similarity scores of all games with the entered game
    similarity_scores = list(enumerate(similarity_scores[index]))

    # Sort games based on similarity score
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Initialize an empty list for recommended game names
    recommendations = []

    # Get the top 5 recommended games
    count = 0
    indices = []
    franchises = set()
    for index, score in similarity_scores:
        game_name = df_filtered.iloc[index]['Name']
        franchise = game_name.split()[0]
        if franchise not in franchises:
            recommendations.append(game_name)
            indices.append(index)
            franchises.add(franchise)
            count += 1
        if count == 5:
            break

    # Create a new dataframe for the recommended games
    df_recommendations = df_filtered.iloc[indices].reset_index(drop=True)

    # Add a new column for similarity scores
    df_recommendations['Similarity Score'] = [score for index, score in similarity_scores if index in indices]

    # Sort the recommendations based on similarity scores
    df_recommendations = df_recommendations.sort_values(by=['Similarity Score'], ascending=False).reset_index(drop=True)

    # Display the recommendations
    if df_recommendations.empty:
        st.write("Sorry, no recommendations found for your preferences.")
    else:
        st.write("\nHere are the top 5 recommended games based on your preferences:\n")
        st.write(df_recommendations[['Name', 'Year', 'Publisher', 'Global_Sales', 'Similarity Score']])