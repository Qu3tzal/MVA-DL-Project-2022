import sklearn


def train_model(model, dataset):
    """ Trains the given model on the given dataset. """
    # Split into train and test sets.
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset[0], dataset[1])
