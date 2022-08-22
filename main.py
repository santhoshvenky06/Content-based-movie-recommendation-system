import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

finaldata = pd.read_csv('C:\Users\santh\Desktop\projects\movie-recomendation\processed movie data\finaldata.csv')

finaldata.head()

finaldata.shape

vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(finaldata['combo'])

similarity = cosine_similarity(feature_vectors)

print(similarity)

print(similarity.shape)

finaldata.tail()

movie_title = input(' Enter your favourite movie name : ')

finaldata.rename(columns = {'Unnamed: 0':'index'}, inplace = True)

movie_list = finaldata['Title'].tolist()
print(movie_list)

find_close_match = difflib.get_close_matches(movie_title, movie_list)
print(find_close_match)

close_match = find_close_match[0]
print(close_match)

index  = finaldata[finaldata.Title == close_match]['index'].values[0]
print(index)

similarity_score = list(enumerate(similarity[index]))
print(similarity_score)

len(similarity_score)

simil_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
print(simil_movies)

print('Movies suggested for you : \n')
i = 1
for movie in simil_movies:
  index = movie[0]
  Title_from_index = finaldata[finaldata.index==index]['Title'].values[0]
  if (i<30):
    print(i, '.',Title_from_index)
    i+=1

movie_name = input(' Enter your favourite movie name : ')

list_of_all_titles = finaldata['Title'].tolist()

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

close_match = find_close_match[0]

index_of_the_movie = finaldata[finaldata.Title == close_match]['index'].values[0]

similarity_score = list(enumerate(similarity[index_of_the_movie]))

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = finaldata[finaldata.index==index]['Title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1


