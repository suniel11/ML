import streamlit as st
import pickle

# Load the similarity matrix and movies DataFrame
similarity = pickle.load(open('similarity.pkl', 'rb'))
movies_df = pickle.load(open('movies.pkl', 'rb'))

# Get list of movie titles
movies_list = movies_df['title'].values

# Define recommendation function
def recommend(movie_name):
    # Find the index of the selected movie
    movie_index = movies_df[movies_df['title'] == movie_name].index[0]
    
    # Get similarity scores for the selected movie
    distances = similarity[movie_index]
    
    # Sort movies by similarity scores (exclude the first one, which is the same movie)
    movie_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    # Get the titles of the recommended movies
    recommend_movies = []
    for i in movie_indices:
        recommend_movies.append(movies_df.iloc[i[0]].title)
    return recommend_movies

# Streamlit app UI
st.title('Movie Recommender')

# Dropdown for movie selection
selected_movie_name = st.selectbox('Select a movie', movies_list)

# Recommend button
if st.button('Recommend'):
    recommendations = recommend(selected_movie_name)
    st.subheader("Recommended Movies:")
    for movie in recommendations:
        st.write(movie)
