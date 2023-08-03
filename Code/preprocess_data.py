from data import read_data, convert_data
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from config import VOCAB_SIZE, MAX_WORDS, TESTING_FILE_PATH, SAVE_PATH
import pickle


def preprocess_data(train_tokenizer=False):
    """
    Functionality to preprocess the data (Tokenization, Padding)
    Not much preprocessing is required since we want to maintain the words as is.

    :return: the training and testing datasets.
    """
    train_df = read_data()
    train_df = convert_data(train_df)

    test_df = read_data(file=TESTING_FILE_PATH)
    test_df = convert_data(test_df)

    # drop null values
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    max_words = MAX_WORDS
    train_df = train_df[train_df["Text"].apply(len) < max_words]

    # tokenize the data
    if train_tokenizer is True:
        tokenizer = Tokenizer(num_words=VOCAB_SIZE)
        tokenizer.fit_on_texts(train_df['Text'])
    else:
        with open(SAVE_PATH + 'tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

    max_len = train_df['Text'].apply(len).max()

    # pad the train and test sequences after fitting on tokenizer
    train_df['Text'] = tokenizer.texts_to_sequences(train_df['Text'])
    X_train = pad_sequences(train_df['Text'],
                            maxlen=max_len,
                            padding='post')
    y_train = train_df['Target']

    test_df['Text'] = tokenizer.texts_to_sequences(test_df['Text'])
    X_test = pad_sequences(test_df['Text'],
                           maxlen=max_len,
                           padding='post')
    y_test = test_df['Target']

    if train_tokenizer is True:
        with open(SAVE_PATH + 'tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return X_train, y_train, X_test, y_test
