import numpy as np
import random
from making_reccomendations import make_10_reccomendations
import pandas as pd

random_users_chosen = random.sample(range(1, 6040), 10)
ratings = pd.read_csv('./data/ratings.csv')


def predictions():
    '''
    Just getting 10 reccomendations for the 10 random users chosen 
    '''
    predictions = []
    for user in random_users_chosen:
        predictions.append(make_10_reccomendations(user_number=str(user)))
    return predictions


def population():
    '''
    Generating a population of the movie_ids and the number of times they have been rated by an unique user
    '''
    no_of_ratings = ratings.groupby('movie_id').count().to_dict(orient='dict')['user_id']
    return no_of_ratings



# Code from the repo https://github.com/statisticianinstilettos/recmetrics
def novelty(predicted, pop: dict, u: int, n: int):
    """
    Computes the novelty for a list of recommendations
    Parameters
    ----------
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    pop: dictionary
        A dictionary of all items alongside of its occurrences counter in the training data
        example: {1198: 893, 1270: 876, 593: 876, 2762: 867}
    u: integer
        The number of users in the training data
    n: integer
        The length of recommended lists per user
    Returns
    ----------
    novelty:
        The novelty of the recommendations in system level
    mean_self_information:
        The novelty of the recommendations in recommended top-N list level
    ----------    
    Metric Defintion:
    Zhou, T., Kuscsik, Z., Liu, J. G., Medo, M., Wakeling, J. R., & Zhang, Y. C. (2010).
    Solving the apparent diversity-accuracy dilemma of recommender systems.
    Proceedings of the National Academy of Sciences, 107(10), 4511-4515.
    """
    mean_self_information = []
    k = 0
    for sublist in predicted:
        self_information = 0
        k += 1
        for i in sublist:
            self_information += np.sum(-np.log2(pop[i]/u))
        mean_self_information.append(self_information/n)
    novelty = sum(mean_self_information)/k
    return novelty, mean_self_information


def calc_novelty():
    '''
    Calculating novelty and the mean novelty from the use of the function above for the 10 users
    chosen at random
    '''
    number_of_ratigs_per_movie = population()
    predictions_for_10_users = predictions()
    nov, mean_nov = novelty(predictions_for_10_users, number_of_ratigs_per_movie,6040,10)
    return nov, mean_nov


if __name__ == '__main__':
    nov, mean_nov = calc_novelty()
    print('The novelty of the recommendations in system level : ', nov)
    print('The novelty of the recommendations in recommended top-N list level : ', mean_nov)