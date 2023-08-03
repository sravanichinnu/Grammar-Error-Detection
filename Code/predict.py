from bi_lstm_model import build_bi_lstm_model
from config import MAX_LEN, SAVE_PATH
import pickle
from keras.utils import pad_sequences


def predict(test_sentences, model=None):
    """
    Functionality to predict the test sentence output.
    :param test_sentences: the sentence to be tested
    :param model: the model on which the prediction is to be made
    :return: the model prediction
    """
    # Default model is the bidirectional lstm model
    if model is None:
        model = build_bi_lstm_model(MAX_LEN)

    # Load the tokenizer
    with open(SAVE_PATH + 'tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Tokenize and pad the test sentences
    test_sentences_tokenized = tokenizer.texts_to_sequences(test_sentences)
    test_sentences_padded = pad_sequences(test_sentences_tokenized,
                                          maxlen=MAX_LEN,
                                          padding='post')

    # Load the model weights
    model.load_weights(SAVE_PATH + model.name)

    # Predict the model output for the test sentence
    predictions = model.predict(test_sentences_padded, verbose=0).round().squeeze()

    return predictions
