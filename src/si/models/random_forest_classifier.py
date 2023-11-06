from tkinter import Radiobutton
import numpy as np
from typing import Literal, Tuple, Union

import numpy as np
import random
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.impurity import gini_impurity, entropy_impurity
from si.models.decision_tree_classifier import DecisionTreeClassifier

class RandomForestClassifier:
    """
    Class representing a random forest classifier.
    """

    def __init__(self, min_sample_split: int = 2, max_depth: int = 10,
                 mode: Literal['gini', 'entropy'] = 'gini', n_estimators: int = 10, max_features: int = None, seed: int = None ) -> None:
        """
        Creates a RandomForestClassifier object.

        Parameters
        ----------
        min_sample_split: int
            minimum number of samples required to split an internal node.
        max_depth: int
            maximum depth of the tree.
        mode: Literal['gini', 'entropy']
            the mode to use for calculating the information gain.
        n_estimators: int
            number of decision trees to use
        max_features: int
            maximum number of features to use per tree
        seed:
            random seed to use to assure reproducibility
        """
        # parameters
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.seed = seed
        
        
        # estimated parameters
        self.trees = []
        
       
    def fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        """
        Fit the trees according to the given training data.

        Parameters
        ----------
        dataset : Dataset
            The training data.

        Returns
        -------
        self : RandomForestClassifier
            The fitted trees.
        """
        self.seed = 1001
        
        # max_features if none
        if self.max_features == None:
            self.max_features = int(np.sqrt(dataset.shape()[1]))
        
        for i in range(self.n_estimators):
            
            num_features = dataset.shape()[1]
            selected_features = random.sample(range(num_features), self.max_features)

            sample_indices = random.choices(range(dataset.shape()[0]), k=dataset.shape()[0])
            
            bootstrap_dataset = Dataset(dataset.X[sample_indices], dataset.y[sample_indices])        
            
            #train tree
            tree = DecisionTreeClassifier(self.min_sample_split, self.max_depth, self.mode)
            fitted_tree = tree.fit(bootstrap_dataset)

            self.trees.append(fitted_tree)
        
        return self
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        dataset : Dataset
            The test data.

        Returns
        -------
        y : array-like, shape = [n_samples]
            The predicted class labels.
        """
  
        def _get_majority_vote(pred: np.ndarray) -> int:
            """
            It returns the majority vote of the given predictions

            Parameters
            ----------
            pred: np.ndarray
                The predictions to get the majority vote of

            Returns
            -------
            majority_vote: int
                The majority vote of the given predictions
            """
            # get the most common label
            labels, counts = np.unique(pred, return_counts=True)
            return labels[np.argmax(counts)]
        
        predictions = np.array([tree.predict(dataset) for tree in self.trees]).transpose()
        return np.apply_along_axis(_get_majority_vote, axis=1, arr=predictions)

    
            
    def score(self, dataset: Dataset) -> float:
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        dataset : Dataset
            The test data.

        Returns
        -------
        score : float
            Mean accuracy
        """
        return accuracy(dataset.y, self.predict(dataset))
        
        
if __name__ == '__main__':
    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
 
    # initialize the Random forest classifier
    RF = RandomForestClassifier(min_sample_split=3, max_depth=3, mode='gini',n_estimators=10)


    RF.fit(dataset_)

    # compute the score
    score = RF.score(dataset_)
    print(f"Score: {score}")
                
        
        
        
        
