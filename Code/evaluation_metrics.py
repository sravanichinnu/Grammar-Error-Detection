import keras.backend as K
import tensorflow as tf


def f1_score(y_true, y_pred):
    """
    Functionality to calculate the F1 Score of the model. The F1 score is a metric used in natural language
    processing to evaluate the performance of a classification model. It is the harmonic mean of the precision and
    recall of the model, with a higher value indicating better performance.

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: f1_score of the predictions
    """
    # Flatten the true and predicted labels
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)

    # Count the number of true positives
    true_positives = K.sum(K.round(K.clip(y_true_flat * y_pred_flat, 0, 1)))

    # Count the number of false positives
    false_positives = K.sum(K.round(K.clip((1 - y_true_flat) * y_pred_flat, 0, 1)))

    # Count the number of false negatives
    false_negatives = K.sum(K.round(K.clip(y_true_flat * (1 - y_pred_flat), 0, 1)))

    # Compute the precision
    precision = true_positives / (true_positives + false_positives + K.epsilon())

    # Compute the recall
    recall = true_positives / (true_positives + false_negatives + K.epsilon())

    # Compute the F1 score
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())

    return f1


def homogeneity(y_true, y_pred):
    """
    Functionality to calculate the homogeneity of the model. Homogeneity is a metric used in natural language
    processing to evaluate the purity of a clustering. It is defined as the ratio of the number of same-label pairs
    to the total number of pairs in the clustering. A higher homogeneity value indicates that the clustering is more
    pure, with a value of 1.0 indicating that all the samples in the clustering have the same label.

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: the homogeneity score of the predictions
    """
    # Flatten the true and predicted labels
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)

    # Compute the number of same-label pairs
    same_label_pairs = K.sum(y_true_flat * y_pred_flat)

    # Compute the total number of pairs
    total_pairs = K.sum(K.ones_like(y_true_flat))

    # Compute the homogeneity
    homogeneity_score = same_label_pairs / (total_pairs + K.epsilon())

    return homogeneity_score


def completeness(y_true, y_pred):
    """
    Functionality to calculate the completeness score of the model. Completeness is a metric used in natural language
    processing to evaluate the completeness of a clustering. It is defined as the ratio of the number of pairs that
    have the same label and are in the same cluster to the total number of pairs with the same label. A higher
    completeness value indicates that the clustering is more complete, with a value of 1.0 indicating that all the
    pairs with the same label are in the same cluster.

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: the completeness score of the predictions.
    """
    # Flatten the true and predicted labels
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)

    # Compute the number of same-label pairs
    same_label_pairs = K.sum(y_true_flat * y_pred_flat)

    # Compute the number of pairs with the same label
    total_same_label_pairs = K.sum(y_true_flat)

    # Compute the completeness
    completeness_score = same_label_pairs / (total_same_label_pairs + K.epsilon())

    return completeness_score


def v_measure(y_true, y_pred):
    """
    Functionality to calculate the v-measure of the model. The V-measure is a metric used in natural language
    processing to evaluate the performance of a clustering. It is defined as the harmonic mean of the homogeneity
    and completeness of the clustering, with a higher value indicating better performance.

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: the v-measure score of the predictions.
    """
    # Compute the homogeneity
    homogeneity_score = homogeneity(y_true, y_pred)

    # Compute the completeness
    completeness_score = completeness(y_true, y_pred)

    # Compute the V-measure
    v = 2 * (homogeneity_score * completeness_score) / (homogeneity_score + completeness_score + K.epsilon())

    return v


def rand_index_loss(y_true, y_pred):
    """
    !!! Not working
    Functionality to calculate the rand index less of the model. The Rand index is a metric that can be used to
    evaluate the performance of a clustering algorithm. It measures the percentage of pairs of data points that are
    either both assigned to the same cluster or both assigned to different clusters by the algorithm and the ground
    truth. In natural language processing, the Rand index can be used to evaluate the performance of clustering
    algorithms on text data.

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: the rand index loss of the predictions.
    """
    # First, get the sets of elements in each group
    true_groups = tf.dynamic_partition(y_true, y_true, 2)
    pred_groups = tf.dynamic_partition(y_pred, y_pred, 2)

    # Then, calculate the Rand index
    rand_index = tf.math.confusion_matrix(true_groups, pred_groups, 2)[1, 1] / tf.size(y_true)

    # Return the Rand index as the loss
    return rand_index


def f05_score(y_true, y_pred):
    """
    Functionality to calculate the F_0.5 score of the model. The F0.5 score is a metric used to evaluate the
    performance of a machine learning model on a classification task. It is a variant of the F1 score, which is a
    weighted average of precision and recall. The F0.5 score gives more weight to precision than the F1 score, and is
    often used when it is more important to avoid false positives than to identify all relevant cases.

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: the F_0.5 score of the predictions
    """
    # Flatten the true and predicted labels
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)

    # Count the number of true positives
    true_positives = K.sum(K.round(K.clip(y_true_flat * y_pred_flat, 0, 1)))

    # Count the number of false positives
    false_positives = K.sum(K.round(K.clip((1 - y_true_flat) * y_pred_flat, 0, 1)))

    # Count the number of false negatives
    false_negatives = K.sum(K.round(K.clip(y_true_flat * (1 - y_pred_flat), 0, 1)))

    # Compute the precision
    precision = true_positives / (true_positives + false_positives + K.epsilon())

    # Compute the recall
    recall = true_positives / (true_positives + false_negatives + K.epsilon())

    # Then, calculate the F0.5 score
    f05 = 1.25 * precision * recall / (0.25 * precision + recall)

    # Return the F0.5 score
    return f05
