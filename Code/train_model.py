from bi_lstm_model import build_bi_lstm_model
from preprocess_data import preprocess_data
from plot_graph import plot_graph
from config import SAVE_PATH


def train_model(evaluation_metrics=None, model=None, train_tokenizer=False):
    """
    Simple test script
    :param evaluation_metrics: the evaluation metrics for the model
    :param model: the model to be evaluated
    :param train_tokenizer: flag to determine whether or not to train the tokenizer
    :return: the history of the model training
    """
    # Get the training and testing data

    X_train, y_train, X_test, y_test = preprocess_data(train_tokenizer)

    if model is None:
        model = build_bi_lstm_model(49)

    if evaluation_metrics is None:
        evaluation_metrics = ['accuracy']

    # Compile the model with loss and optimizers and custom evaluation metrics
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=evaluation_metrics)

    # Store the history of the training data
    history = model.fit(X_train, y_train,
                        batch_size=64,
                        epochs=10,
                        validation_split=0.2)

    # Plot the training and validation metrics
    plot_graph(history)

    # Test the model on the testing set
    test_metrics = model.evaluate(X_test, y_test)

    # Save the model
    model.save(SAVE_PATH + model.name)

    return history, test_metrics
