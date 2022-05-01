import matplotlib.pyplot as plt
import numpy as np
import pickle


def generate(report_data):
    n_min = np.argmin(report_data['val_loss'])


    x = np.arange(0, 5, 0.1)
    y = np.sin(x)
    fig, ax = plt.subplots(nrows = 2, ncols = 2, sharex=True)

    fig.suptitle('Training Report', fontsize = 16)



    ax[0][0].plot(report_data['loss'], 'r')
    ax[0][0].set_title('Train')
    ax[0][0].set_ylabel('Loss')


    ax[0][1].plot(report_data['val_loss'], 'r')
    ax[0][1].set_title('Test')


    ax[1][0].plot(report_data['accuracy'], 'b')
    ax[1][0].set_xlabel('n')
    ax[1][0].set_ylabel('Accuracy')


    ax[1][1].plot(report_data['accuracy'], 'b')
    ax[1][1].set_xlabel('n')


    for i in ax:
        for j in i:
            j.axvline(n_min, color = 'k', linestyle = '--')

    fig.tight_layout()
    plt.savefig('training_report.png')
    plt.show()