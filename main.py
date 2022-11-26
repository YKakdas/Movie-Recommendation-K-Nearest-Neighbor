import sys

import pandas
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

'''
    To test the script, I prepared some predefined function calls in the main. You may just comment out and use them,
    or just can change the parameters as you wish.

'''

''' 
    This function finds recommendations based on given parameters using K-NearestNeighbors regression model.
    
    ------ Parameters ------
    
    keyword : string
    This keyword might be the full name of the movie or just a substring of the movie. For instance, if a user enters 
    "Harry Potter and the Chamber of Secrets (2002)", algorithm will only find recommendations based on this specific
    movie. However, if the user enters only "Harry Potter", algorithm will find all movies that include this keyword, in
    this case 8 movies of Harry Potter series will be found and recommendation will be based on all this found movies.
    
    number_of_recommendations : integer
    Determines how many neighbors we should find for given keyword to recommend
    
    show_only_best_results : boolean If given keyword is a substring of more than one movies, if this parameter is 
    True, algorithm will find list with size of number_of_recommendations, which is closest distances of all 
    neighbors of found movies by the keyword. If this parameter is false, algorithm will return a list with 
    number_of_recommendations for each found movie. Therefore, if keyword is substring of k movies, algorithm will 
    return total k * number_of_recommendations 
    
    exclude_found_movie : boolean
    Since found movie by given keyword is the center of the clusters, this parameter determines whether we should
    add initial movies to the list or not. Let's say given keyword is "Harry Potter", in case we set false to this
    parameter, algorithm will include all Harry Potter movies to the recommendation as well since it is the substring of
    all Harry Potter movies. If it is set to true, then algorithm will find number_of_recommendations by excluding the
    initial found movie by keyword.#
'''


def find_recommendations(keyword, number_of_recommendations, show_only_best_results, exclude_found_movie):
    #   for debugging purposes, output settings of pandas
    #   pd.set_option('display.max_rows', None)
    #   pd.set_option('display.max_columns', None)

    # attribute names of movies.csv
    movies_column_names = ['movieId', 'title', 'genres']

    # attribute names of ratings.csv
    ratings_column_names = ['userId', 'movieId', 'rating', 'timestamp']

    # read movies and ratings into dataframe of pandas
    movies = pd.read_csv("movies.csv", names=movies_column_names)
    ratings = pd.read_csv("ratings.csv", names=ratings_column_names)

    # making ratings dataframe easier to be processed and read by creating table where
    # rows are movie Ids and columns corresponding userId.
    ratings_pivoted = ratings.pivot(index='movieId', columns='userId', values='rating')

    # for more accurate results remove noise

    # axis=0 means count rows which counts number of ratings for each movieId. If movie doesn't have
    # more than 20 voter, delete that entry. I have decided 20 by intuitively
    df = ratings_pivoted.loc[:, (ratings_pivoted.notnull().sum(axis=0) > 20)]

    # axis=1 means count columns which counts number of movies voted by each user. If a user hasn't rated more than
    # 20 movies, do not consider that person's ratings as recommendation and remove that entry. I have decided 20 by
    # intuitively
    df = df.loc[(ratings_pivoted.count(axis=1) > 20), :]

    # If a user hasn't voted a movie, or a movie hasn't been voted by a user, that entry is NaN.
    # Replace null values by 0.0 to allow KNN work properly.
    df.fillna(0, inplace=True)

    # Reset index to original so that we can use indexing by column attribute names
    df.reset_index(inplace=True)

    # Since every user doesn't vote every movie, our input dataset is highly sparse matrix. Therefore, use
    # scipy.sparse.csr_matrix to compress the sparse matrix into smaller dense matrix. It will give us performance
    # benefits.
    dense_input_matrix = csr_matrix(df.values)

    # I've used metric "cosine" and algorithm = auto.
    # n_jobs=-1 means use all available processors to parallelize the algorithm
    knn = NearestNeighbors(metric="cosine", n_neighbors=20, n_jobs=-1)

    # instead of using our original sparse matrix, use compacted matrix for performance concerns
    knn.fit(dense_input_matrix)

    # find all movies that contain the given keyword in itself
    movie_list = movies[movies['title'].str.contains(keyword)]

    # If found movie was part of the deleted noised data, ignore that movie
    count = 0
    drop_indices = []
    for index in range(len(movie_list)):
        movie_id = movie_list.iloc[index]['movieId']
        if movie_id not in df['movieId'].values:
            drop_indices.append(movie_list[movie_list['movieId'] == movie_id].index)
        count += 1
    for index in drop_indices:
        movie_list = movie_list.drop(index)

    # If there is no matched movie with given keyword, show error and exit the application
    if len(movie_list) == 0:
        print("Could not find any recommendation with the given keyword", keyword)
        sys.exit()

    # Print initially found movies to be used as cluster center for recommendation
    print("----------------------------------------------------------------------------->")
    print(len(movie_list), "movies has been found for the given keyword", keyword)
    print("Movie recommendation will be based on the following found movies...")
    print_movies(movie_list)
    print("----------------------------------------------------------------------------->")

    movies_to_be_recommended_all = []
    all_recommended_movies_dataframe: pandas.DataFrame

    number_of_movies_processed = 0
    for index in range(len(movie_list)):
        movies_to_be_recommended_single = []
        single_recommended_movies_dataframe: pandas.DataFrame

        # fetch movieId from the movie list
        movie_id = movie_list.iloc[index]['movieId']

        # This one returns row of the table which includes all user ratings for given movieId
        ratings = df[df['movieId'] == movie_id]

        # if dataframe is empty, continue looping the others
        if ratings.empty:
            continue
        else:
            number_of_movies_processed += 1

        # Find corresponding userId for mapping entry with the dense matrix
        user_id = ratings.index[0]

        # fetch number_of_recommendations from neighbors for given cluster center point, which is the reference movie.
        # Also, keep track of distances for filtering
        distances, indices = knn.kneighbors(dense_input_matrix[user_id], n_neighbors=number_of_recommendations + 1)

        # For each found neighbor, extract data from dataframe and put into a list to print later
        count = 0
        for val in indices[0]:
            movie_id = df.iloc[val]['movieId']
            idx = movies[movies['movieId'] == movie_id].index
            movies_to_be_recommended_single.append(
                {'Title': movies.iloc[idx]['title'].values[0], 'Genres': movies.iloc[idx]['genres'].values[0],
                 'Distance': distances[0][count]})
            count = count + 1
        single_recommended_movies_dataframe = pd.DataFrame(movies_to_be_recommended_single,
                                                           columns=['Title', 'Genres', 'Distance'])

        # If user only wants to see best results, save all found movies to filter later. If not, print
        # number_of_recommendation movies for reference the reference movie immediately
        if not show_only_best_results:
            print_recommendations(single_recommended_movies_dataframe, exclude_found_movie, number_of_recommendations,
                                  show_only_best_results, keyword)
        else:
            movies_to_be_recommended_all.extend(movies_to_be_recommended_single)

    # do filter for the best results. Sort by distance returned by knn, find the closest neighbors
    if show_only_best_results:
        all_recommended_movies_dataframe = pd.DataFrame(movies_to_be_recommended_all,
                                                        columns=['Title', 'Genres', 'Distance'])
        all_recommended_movies_dataframe = all_recommended_movies_dataframe.sort_values(by=['Distance'])
        all_recommended_movies_dataframe = all_recommended_movies_dataframe.drop_duplicates(subset=['Title'])
        print_recommendations(all_recommended_movies_dataframe, exclude_found_movie, number_of_recommendations,
                              show_only_best_results, keyword)


def print_movies(movie_list):
    count = 1
    for index, row in movie_list.iterrows():
        print(count.__str__() + ".", "Movie title :", row['title'], "Genres :", row['genres'])
        count += 1


def print_recommendations(movie_list, exclude_found_movie, number_of_recommendations, show_only_best_results, keyword):
    movie_list = movie_list.sort_values(by=['Distance'])
    if show_only_best_results:
        print("Recommendations based on the given keyword", keyword)
    else:
        print("Recommendations based on the movie Title:", movie_list.iloc[0]['Title'], "Genres :",
              movie_list.iloc[0]['Genres'])
    if exclude_found_movie:
        if show_only_best_results:
            index_names = movie_list[movie_list['Distance'] < 0.000000001].index
            movie_list = movie_list.drop(index_names)
        else:
            movie_list = movie_list.drop(0)
    else:
        movie_list = movie_list.drop(movie_list.tail(1).index)
    print("----------------------------------------------------------------------------->")
    count = 1
    for index, row in movie_list.iterrows():
        print(count.__str__() + ".", "Movie title :", row['Title'], "Genres :", row['Genres'])
        count += 1
        if count == number_of_recommendations + 1:
            return
    print("----------------------------------------------------------------------------->")


if __name__ == '__main__':
    find_recommendations(keyword='Aven', number_of_recommendations=20, show_only_best_results=True,
                         exclude_found_movie=False)
#    find_recommendations(keyword='Aven', number_of_recommendations=10, show_only_best_results=False,
#                         exclude_found_movie=False)
#    find_recommendations(keyword='Aven', number_of_recommendations=10, show_only_best_results=False,
#                         exclude_found_movie=True)
#    find_recommendations(keyword='Billy Madison', number_of_recommendations=10, show_only_best_results=True,
#                         exclude_found_movie=True)
#    find_recommendations(keyword='Interstellar', number_of_recommendations=10, show_only_best_results=True,
#                         exclude_found_movie=False)
#    find_recommendations(keyword='This one is for seeing the case when no movies has been found',
#                         number_of_recommendations=10, show_only_best_results=False, exclude_found_movie=False)
