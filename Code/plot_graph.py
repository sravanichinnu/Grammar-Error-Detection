import matplotlib.pyplot as plt


def plot_graph(history):
    for metric in history.history:
        if 'val_' in metric:
            continue
        try:
            # Get the training scores for the metric
            train_scores = history.history[metric]

            # Get the validation scores for the metric
            val_metric = 'val_' + metric
            val_scores = history.history[val_metric]

            # Get the number of epochs
            epochs = range(1, len(train_scores) + 1)

            # Plot the metrics
            plt.plot(epochs, train_scores, label=metric)
            plt.plot(epochs, val_scores, label=val_metric)

            # Give labels to the X and Y axis
            plt.xlabel('Epochs')
            plt.ylabel(metric)

            # Plot the legend
            plt.legend()

            # Give a title to the plot
            plt.title(f"{metric} vs epochs")

            plt.show()

            print()

        except Exception:
            continue
