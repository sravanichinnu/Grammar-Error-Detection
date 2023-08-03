from predict import predict


def run_demo(models, test_sentences):
    """
    Demo script to test the pretrained models.
    :param models: the list of models on which the prediction is to be made
    :param test_sentences: the test sentences
    :return: None, prints the predictions of each model on the given test sentences
    """
    for sentence in test_sentences:
        print('\n\nPrediction for sentence: ', sentence)
        for model in models:
            print(model.name, ' -> ', predict(test_sentences=[sentence], model=model))
