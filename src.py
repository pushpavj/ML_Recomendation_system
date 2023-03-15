import numpy as np
import pandas as pd

# Get the Movie data set from kaggle. This data set contains all the details
#  of the each movie, example,
# Movie id, director, cast, key words, movie name, languages…..each and 
# every thing in it.

movies = pd.read_csv('data/tmdb_5000_movies.csv')
credits = pd.read_csv('data/tmdb_5000_credits.csv')

print("movies.head():",movies.head())
print("credits.head():",credits.head())

print("movies.shape:",movies.shape)
print("credits.shape:",credits.shape)

#Concatinate the movie
movies=movies.merge(credits,on='title')

print("movies.head():",movies.head())
print("movies.shape:",movies.shape)

# Keeping important columns for recommendation
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
print("movies.head():",movies.head())
print("movies.shape:",movies.shape)

#Handle null values
print("movies null values:",movies.isnull().sum())
print("movies dropna:",movies.dropna(inplace=True))

print("movies.head():",movies.head())
print("movies.shape:",movies.shape)

#Check if there are any duplicates
print("movies duplicates:",movies.duplicated())

#Handling geners
print("movies.genres:",movies.genres[0])

import ast #for converting str to list
#The ast module helps Python applications to process trees of the Python
#  abstract syntax grammar. The abstract syntax itself might change with 
# each Python release; this module helps to find out programmatically what 
# the current grammar looks like and allows modifications of it.
def convert(text):
    """
    This function will take a string and convert it to a list.
    """
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L

#Geners has dictionary values, we take only the name values
movies['genres'] = movies['genres'].apply(convert)

print("movies.genres:",movies.genres[0])

# handle keywords, we take only the name values
print("movies.iloc[0]['keywords']",movies.iloc[0]['keywords'])
movies['keywords'] = movies['keywords'].apply(convert)
print("movies.head():",movies.head())

# handle cast.It has multilple dictionaries, we take only the name values of first 3 dictionaries
print("movies.iloc[0]['cast']",movies.iloc[0]['cast'])

# Here i am just keeping top 3 cast

def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L
movies['cast'] = movies['cast'].apply(convert_cast)
print("movies.head()",movies.head())

# handle crew, we take only the name whose job is director 

print("movies.iloc[0]['crew']",movies.iloc[0]['crew'])

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L
movies['crew'] = movies['crew'].apply(fetch_director)

print("movies.head():",movies.head())

# handle overview (converting single string to list of words)

print("movies.iloc[0]['overview']",movies.iloc[0]['overview'])

movies['overview'] = movies['overview'].apply(lambda x:x.split()) # split will split the string into 
                                                    # a list of words


print("movies.sample()",movies.sample(4))
print("movies.iloc[0]['overview']",movies.iloc[0]['overview'])


# now removing space like that 
#Remove the space between two words
'Anna Kendrick'
'AnnaKendrick'

def remove_space(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew'].apply(remove_space)
movies['genres'] = movies['genres'].apply(remove_space)
movies['keywords'] = movies['keywords'].apply(remove_space)

print("movies.head():",movies.head())


# Concatinate all the words in to a single list and store it under a tags column
#The tags column is a list of strings, each string is a word in the particular movie.
# i.e. combining actor, director, genre, keywords, and overview into a single list as a bag of words
# of the movie.
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

print("movies.head():",movies.head())

print("movies.iloc[0]['tags']",movies.iloc[0]['tags'])

# droping those extra columns and keep only movie id, title and tags details
new_df = movies[['movie_id','title','tags']]

print("new_df.head():",new_df.head())

# Converting tags list to single str
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
print("new_df.head():",new_df.head())

# Converting all letters in the tags to lower case
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

print("new_df.head():",new_df.head())

print("new_df.iloc[0]['tags']",new_df.iloc[0]['tags'])
import nltk
from nltk.stem import PorterStemmer

ps = PorterStemmer() #Porter stemmer is a class that is used to stem the words in a list of words
#stemming means removing the punctuations, numbers, and symbols from the words in the list

def stems(text):
    T = []
    
    for i in text.split():
        T.append(ps.stem(i)) #stemming will remove all non-alphanumeric characters
        # stemming will help us to find the most common words in the text
        
    
    return " ".join(T)

new_df['tags'] = new_df['tags'].apply(stems)

print("new_df.iloc[0]['tags']",new_df.iloc[0]['tags'])

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english') # stop_words is a list of words that 
# will be removed such as is, the, on, ....etc
#Max_features is the maximum number of words that will be extracted from the text
#

#Vectorizing the text means converting the text into to numerical values
vector = cv.fit_transform(new_df['tags']).toarray()
#How does the vector generate?
#The CountVectorizer class is used to generate a matrix of features for the given number of features.
#It makes the list of features from all the records in the data set.
# Then for each record, it will compare the word in the record with the word in the features matrix.
# If the word in the record is found in the features matrix, it will give the count of the word in the
# features matrix. if any word in the feature matrix is not found in the record, it will give 0.
# This way a numeric array of features can be generated with the length same as that of the features
# matrix.
print("vecotor[0]",vector[0])

print("vector.shape:",vector.shape)

print("len(cv.get_feature_names())",len(cv.get_feature_names()))

from sklearn.metrics.pairwise import cosine_similarity
#Cosine similarity is a measure of similarity between two vector. Using the numerical array of 
# features we can draw a vector or line from it. When you draw line for all the records in the data set, 
# and if you measure the angle between the two vectors, it will give the cosine similarity between 
# the two vectors. 
similarity = cosine_similarity(vector)

print("similarity.shape",similarity.shape)
print("similarity[0][0]",similarity[0][0])

print("new_df.head()",new_df.head())


new_df[new_df['title'] == 'The Lego Movie'].index[0] #gives the index of the record which has the title
                                         # as The Lego Movie
print("new_df[new_df['title'] == 'The Lego Movie']",new_df[new_df['title'] == 'The Lego Movie'])
print("new_df[new_df['title'] == 'The Lego Movie'].index",new_df[new_df['title'] == 'The Lego Movie'].index)
def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0] #gets index of the record which has the title 
                                                        # as movie
    print("index:",index)
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    print("similarity[index]",similarity[index])
    print("length of similarity:",len(similarity))
    
    #Similarity gives the distance between the movie and each the record vector for all the 
    # records in the data set.
    #enumerate is used to get the index of each record                                         #in the data set
    #The distance is a list of tuples, each tuple contains the index of the record in the data set
    # and the cosine similarity between the two vectors.
    #This will find the similarity distance between the given index with the each record in the data 
    # set and return the index of the record which has the highest cosine similarity at first followed
    # by rest of the records in the data set.

   # print("distances:",distances)
    print("distance length",len(distances))
    for i in distances[1:6]: #once we have the index of the top 6 records in the data set which has
             # the heighest cosine similarity for the given index, now we will display the movie title
             # of those records as the recommended movie.
        print(new_df.iloc[i[0]].title)

recommend('Spider-Man 2')

#Save the data set as well as the similarity model as a pickle file
import pickle

pickle.dump(new_df,open('artifacts/movie_list.pkl','wb'))
pickle.dump(similarity,open('artifacts/similarity.pkl','wb'))







