import itertools
from typing import Callable, Tuple, Dict, Any

import numpy as np

from si.data.dataset import Dataset
from si.model_selection.cross_validation import k_fold_cross_validation

def randomized_search_cv(model,
                   dataset: Dataset,
                   hyperparameter_grid: Dict[str, Tuple],
                   scoring: Callable = None,
                   cv: int = 5,
                   n_iter: int = 10) -> Dict[str, Any]:
    """
    Performs a randomize search cross validation on a model.

    Parameters
    ----------
    model
        The model to cross validate.
    dataset: Dataset
        The dataset to cross validate on.
    hyperparameter_grid: Dict[str, Tuple]
        The hyperparameter grid to use.
    scoring: Callable
        The scoring function to use.
    cv: int
        The cross validation folds.

    Returns
    -------
    results: Dict[str, Any]
        The results of the randomized search cross validation. Includes the scores, hyperparameters,
        best hyperparameters and best score.
    """
    # validate the parameter grid
    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")
        
    results = {'scores': [], 'hyperparameters': []}
    
    # generate list of combinations
    parameter_names = list(hyperparameter_grid.keys())
    parameter_values = list(hyperparameter_grid.values())
    
    random_combinations = []

    for _ in range(n_iter):
        random_values = [np.random.choice(param) for param in parameter_values]
        random_combinations.append(random_values)

    # for each combination
    for combination in random_combinations:

        # parameter configuration
        parameters = {}

        # set the parameters
        for parameter, value in zip(hyperparameter_grid.keys(), combination):
            setattr(model, parameter, value)
            parameters[parameter] = value

        # cross validate the model
        score = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        # add the score
        results['scores'].append(np.mean(score))

        # add the hyperparameters
        results['hyperparameters'].append(parameters)

    results['best_hyperparameters'] = results['hyperparameters'][np.argmax(results['scores'])]
    results['best_score'] = np.max(results['scores'])
    return results

if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.models.logistic_regression import LogisticRegression

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)

    # initialize the Logistic Regression model
    knn = LogisticRegression()

    # parameter grid
    parameter_grid_ = {
        'l2_penalty': (1, 10),
        'alpha': (0.001, 0.0001),
        'max_iter': (1000, 2000)
    }

    # cross validate the model
    results_ = randomized_search_cv(knn,
                              dataset_,
                              hyperparameter_grid=parameter_grid_,
                              cv=3,
                              n_iter=12)

    # print the results
    print(results_)

    # get the best hyperparameters
    best_hyperparameters = results_['best_hyperparameters']
    print(f"Best hyperparameters: {best_hyperparameters}")

    # get the best score
    best_score = results_['best_score']
    print(f"Best score: {best_score}")