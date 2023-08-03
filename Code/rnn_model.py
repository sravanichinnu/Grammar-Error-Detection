from keras.models import Sequential
from keras.layers import Embedding, Dense, SimpleRNN
from config import VOCAB_SIZE, MAX_LEN


def build_rnn_model(max_len=MAX_LEN, embedding_len=128, rnn_units=[4,40,40]):
    """
    Functionality to build a simple bidirectional rnn model with 64 rnn units.

    :param max_len: maximum input length of the sequences
    :param embedding_len: the dimension of the output of the embedding layer
    :param rnn_units: the number of rnn units required
    :return: the rnn model
    """

    model = Sequential(name='rnn_model')
    model.add(Embedding(input_dim=VOCAB_SIZE,
                        output_dim=embedding_len,
                        input_length=max_len))
    for units in rnn_units[:-1]:
        model.add(SimpleRNN(units, return_sequences=True, activation='tanh', dropout=0.15))
    model.add(SimpleRNN(rnn_units[-1], activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    return model
