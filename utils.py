import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_scores(scores, title, fname, show=False, savefig=False):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.title(title)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    if savefig:
        plt.savefig(fname)
    if show:
        plt.show()
    plt.close()


def save_checkpoint(state, filename):
    """
    Save state to filename
    :param state:
    :param filename:
    :return:
    """
    torch.save(state, filename)
