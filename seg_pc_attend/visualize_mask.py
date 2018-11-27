import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    fig = plt.figure(1)
    accs = [0.211, 0.824, .820, .815, .907, .892, .842, .818, .807, .799, .793]
    masks = np.arange(11)
    masks = masks / 10

    plt.plot(masks, accs)
    plt.title('Test Performance vs. Point Cloud Masking Rate')
    plt.xlabel('Masking Rate')
    plt.ylabel('Accuracy')
    # plt.ylim(40, 100)
    # plt.yticks(np.arange(40, 100, 5))
    plt.show()




