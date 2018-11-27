import matplotlib.pyplot as plt
import numpy as np



if __name__ == '__main__':
    log_file = 'deep_sets_seg_cpu_train.log'
    file = open(log_file)
    training = {'train_loss': [],
                'train_acc': [],
                'zeros_acc': [],
                'ones_acc': [],
                'val_loss': [],
                'val_acc': [],
                'val_zeros_acc': [],
                'val_ones_acc': []}

    for line in file:
        if (line[0] == 't'):
            parse_line = line.split('-')
            train = [i.split(':') for i in parse_line]
            train_loss = float(train[0][1].strip())
            train_acc = float(train[1][1].strip())
            zeros_acc = float(train[2][1].strip())
            ones_acc = float(train[3][1][:-1].strip())

            training['train_loss'].append(train_loss)
            training['train_acc'].append(train_acc)
            training['zeros_acc'].append(zeros_acc)
            training['ones_acc'].append(ones_acc)
        elif (line[0] == '-'):
            parse_line = line.split('-')
            val = [i.split(':') for i in parse_line]
            val_loss = float(val[1][1].strip())
            val_acc = float(val[2][1].strip())
            val_zeros_acc = float(val[3][1].strip())

            val_ones_acc = val[4][1][:-1].strip()
            if (val_ones_acc[-1] == ']'):
                val_ones_acc = float(val_ones_acc.split(' ')[0].strip())
            else:
                val_ones_acc = float(val_ones_acc)

            training['val_loss'].append(val_loss)
            training['val_acc'].append(val_acc)
            training['val_zeros_acc'].append(val_zeros_acc)
            training['val_ones_acc'].append(val_ones_acc)
        else:
            continue

    file.close()

    num_epochs = len(training['train_loss'])
    x = np.arange(1, num_epochs + 1)

    fig = plt.figure(1)
    plt.plot(x, training['train_loss'])
    plt.plot(x, training['val_loss'])
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yticks()
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.show()

    fig = plt.figure(2)
    plt.plot(x, training['train_acc'])
    plt.plot(x, training['val_acc'])
    plt.plot(x, training['val_ones_acc'])
    plt.plot(x, training['val_zeros_acc'])
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(40, 100)
    plt.yticks(np.arange(40, 100, 5))
    plt.legend(['Train Accuracy', 'Validation Accuracy',
                'Validation Foreground Baseline', 'Validation Background Baseline'])
    plt.show()




