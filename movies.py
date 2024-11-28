import pandas as pd  
import numpy as np  
from sklearn.feature_extraction.text import TfidfVectorizer  # For text processing
from sklearn.metrics.pairwise import cosine_similarity  # To compute similarity
import ast
from nltk.stem.porter import PorterStemmer

# Load dataset
movies = pd.read_csv('movies.csv')  # Replace 'movies.csv' with your file path
credits = pd.read_csv('tmdb_5000_credits.csv')


movies =movies.merge(credits, on='title' )


# columns to work on :  genres , id , keywords, title , overview ,  title , cast , crew.director

movies = movies[['id' , 'title' , 'overview' , 'genres' , 'keywords' , 'cast' , 'crew'] ]


# The `ast` module helps Python applications to process trees of the Python
# Evaluate an expression node or a string containing only a Python


# storing the names in  list from a given onject
def convert(obj):
        L = []
        for i in ast.literal_eval(obj):
                L.append(i['name'])
        return L


movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# getting the top/first three actors from the cast
def convert3(obj):
        L = []
        counter = 0 
        for i in ast.literal_eval(obj):
            if counter != 3:
                L.append(i['name'])
                counter +=1
            else:
              break
        return L

movies['cast'] = movies['cast'].apply(convert3)


def fetch_director(obj):
        L = []
        
        for i in ast.literal_eval(obj):
            if  i['job'] == 'Director':
                L.append(i['name'])
               
             
                break
        return L



movies['crew'] = movies['crew'].apply(fetch_director)


#
movies['overview'] = movies['overview'].fillna('')
movies['overview'] = movies['overview'].apply(lambda x:x.split())
# print (movies.head()['overview'])

# deleteing the spacia bw the words to avoid confusion


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" " , "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" " , "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" " , "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" " , "") for i in x])
# print (movies.head())
# print (movies.head())

# making a new coloum to provide ease in going through the data
movies['tag'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['crew']

# print (movies.head()['tag'])

# storing the data in new dataset
new_df = movies[['id' , 'title' , 'tag']]

new_df['tag'] = new_df['tag'].apply(lambda x :" ".join(x).lower())
# print (new_df.head()['tag'][0])


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000,stop_words='english')
vectors = cv.fit_transform(new_df['tag'])
# print(vectors.shape)
vectors = vectors.toarray()

# print(vectors[0])
# print(cv.get_feature_names_out())

ps = PorterStemmer()

def stem(text):
      y =[]

      for i in text.split():
            y.append(ps.stem(i))

      return " ".join(y)



new_df['tag'] = new_df['tag'].apply(stem)
# print(new_df.head()['tag'])


from sklearn.metrics.pairwise import cosine_similarity


similarity = cosine_similarity(vectors)

# print(similarity[0])

def recommend(movie):
      movie_index = new_df[new_df['title'] == movie].index[0]
      distances = similarity[movie_index]
      movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda X:X[1])[1:6]

      for i in movies_list:
            print(new_df.iloc[i[0]].title)
    
    
      return



# newtitle = new_df.iloc[3333].title
# print(newtitle)
recommend('The Wave')


import pickle 

pickle.dump(new_df, open('movies.pkl' , 'wb'))

pickle.dump(similarity , open('similarity.pkl' , 'wb'))