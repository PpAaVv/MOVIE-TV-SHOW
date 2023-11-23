import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Load the MovieLens dataset (you can download it from https://grouplens.org/datasets/movielens/)
# I'm assuming you have a CSV file with columns: userId, movieId, rating
# Adjust the file path accordingly.
data = pd.read_csv('path/to/movielens/dataset.csv')

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create a user-item matrix
user_movie_matrix = train_data.pivot_table(index='userId', columns='movieId', values='rating')

# Fill missing values with 0
user_movie_matrix = user_movie_matrix.fillna(0)

# Calculate the cosine similarity between users
user_similarity = cosine_similarity(user_movie_matrix)

# Convert the similarity matrix into a DataFrame
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# Function to get movie recommendations
def get_movie_recommendations(user_id):
    # Get the movies that the user has not rated
    user_not_rated = user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] == 0].index

    # Calculate the predicted ratings for the unrated movies
    predicted_ratings = user_similarity_df.loc[user_id] @ user_movie_matrix.loc[:, user_not_rated]

    # Sort the movies based on predicted ratings in descending order
    recommended_movies = predicted_ratings.sort_values(ascending=False)

    return recommended_movies.index

# Example: Get movie recommendations for user with ID 1
user_id = 1
recommendations = get_movie_recommendations(user_id)

print(f"Top 5 movie recommendations for user {user_id}:")
print(recommendations[:5])




