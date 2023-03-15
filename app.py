'''
Author: Pushpa 
Email: pushpapraketh@gmail.com
Date: 2023-Mar-15
'''

import pickle
import streamlit as st
import requests

def fetch_poster(movie_id):
    """
    This function will first gets the json data from the movie API for the
    given movie_id. Then it will use the json data to fetch the poster path.
    From the poster path we will get the poster image.
      """
    #pass the movie id and get the poster url data
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path'] #get the poster path
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    # form the full path of the image of the movie poster
    return full_path

def recommend(movie):
    #gets index of the record which has the title as movie
    index = movies[movies['title'] == movie].index[0]
#The distance is a list of tuples, containing the index of the
# record and the cosine similarity between the given index with the
# records in the data set.
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)
  #gives top 5 recommended movies details
    return recommended_movie_names,recommended_movie_posters

#Build the app frame work using Streamlit
st.header('Movie Recommender App built by Praketh') 
#write the title of the app
#Dowoload the movie list
movies = pickle.load(open('artifacts/movie_list.pkl','rb'))
#download the similarity matrix model built earlier
similarity = pickle.load(open('artifacts/similarity.pkl','rb'))
#get the list of movie names
movie_list = movies['title'].values
#build the drop down cell with the list of movie names
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'): #build the button to show the 
    #recommended movies
    recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5) #build 5 columns
    # and name the columns
    with col1:
        st.text(recommended_movie_names[0]) #display the first movie name
        st.image(recommended_movie_posters[0])#display the first movie poster
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])

