from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM

from config import VOCAB_SIZE, MAX_LEN


def build_lstm_model(max_len=MAX_LEN, embedding_len=128, lstm_units=[64, 128, 128]):
    """
    Functionality to build a simple lstm model with multiple layers lstm units.

    :param max_len: maximum input length of the sequences
    :param embedding_len: the dimension of the output of the embedding layer
    :param lstm_units: the number of lstm units required for the model
    :return: the lstm model
    """

    model = Sequential(name='lstm_model')
    model.add(Embedding(input_dim=VOCAB_SIZE,
                        output_dim=embedding_len,
                        input_length=max_len))
    for units in lstm_units[:-1]:
        model.add(LSTM(units, return_sequences=True, dropout=0.15))
    model.add(LSTM(lstm_units[-1]))
    model.add(Dense(1, activation='sigmoid'))

    return model
