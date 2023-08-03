import pandas as pd
from config import TRAINING_FILE_PATH


def convert_data(df):
    """
    Functionality to convert the dataset of incorrect and correct sentences to a dataset of text and target.
    Here the text is the mixture of correct and incorrect sentences and the target whether the sentence is
    grammatically correct (0) or incorrect (1).

    :param df: dataframe on incorrect and their correct sentences
    :return: the combined dataset with their target variables
    """
    # Storing the correct text
    df_correct = pd.DataFrame({
        "Text": [],
        "Target": []
    })
    df_correct['Text'] = df[1]
    df_correct['Target'] = 0

    # Storing the incorrect text
    df_incorrect = pd.DataFrame({
        "Text": [],
        "Target": []
    })
    df_incorrect['Text'] = df[0]
    df_incorrect['Target'] = 1

    # Creating a single dataframe of correct and incorrect sentences.
    df_new = pd.concat([df_correct, df_incorrect])
    return df_new


def read_data(file=TRAINING_FILE_PATH):
    """
    Functionality to return the data from the file.

    :return: the dataset of incorrect and their corresponding correct sentences.
    """
    df = pd.read_csv(file,
                     delimiter=".\t",
                     header=None,
                     engine='python')
    return df
