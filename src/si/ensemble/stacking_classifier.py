import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy

class StackingClassifier():
    
    def __init__(self,models, final_model):
        """
        Initialize the ensemble classifier.

        Parameters
        ----------
        models: array-like, shape = [n_models]
            Different models for the ensemble.
        final_model: 
            Final model chosen

        """
        # parameters
        self.models = models
        self.final_model = final_model
    
    def fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Fit the models according to the given training data.

        Parameters
        ----------
        dataset : Dataset
            The training data.

        Returns
        -------
        self : StackingClassifier
            The fitted model.
        """
        for model in self.models:
            model.fit(dataset)

        models_predictions = np.array([model.predict(dataset) for model in self.models]).transpose()
        n_dataset = Dataset(models_predictions, dataset.y)
        
        #train the final model
        self.final_model.fit(n_dataset)
        
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
        n_dataset = self.fit(dataset)
        results = self.final_model.predict(n_dataset)
        return results

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
    # import dataset
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split
    from si.models.knn_classifier import KNNClassifier
    from si.models.logistic_regression import LogisticRegression

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # initialize the KNN and Logistic classifier
    knn = KNNClassifier(k=3)
    lg = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)
    knn_2 = KNNClassifier(k=3)
    
    # initialize the Voting classifier
    stacking = StackingClassifier([knn, lg], knn_2)

    stacking.fit(dataset_train)

    # compute the score
    score = stacking.score(dataset_test)
    print(f"Score: {score}")

    print(stacking.predict(dataset_test))
