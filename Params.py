class Params(object):
    bandwidth = 8                                               #Glimpse window size
    # bandwidth = win_size ** 2                                 #Sensor bandwidth
    batch_size = 32                                             #Training batch size
    eval_batch_size = 50                                        #Evaluation batch size
    loc_std = 0.22                                              #Std when setting location
    original_size = 28                                          #Original size of input image (MNIST is 28x28)
    num_channels = 1                                            #Number of channels in input image (MNIST is greyscale)
    depth = 1                                                   #Scale of foveal glimpse (i.e. how many decreasing resolution windows used
    sensor_size = bandwidth ** 2 * depth * num_channels         #Total size of sensor
    minRadius = 8                                               #Minimum radius to consider in glimpse
    hg_size = hl_size = 128                                     #Dimensionality of glimpse and location representations in glimpse network
    g_size = 256                                                #Dimensionality of glimpse network
    cell_output_size = 256                                      #Dimensionality of glimpse network output
    cell_core_size = 256                                        #Dimensionality of core LSTM network
    loc_dim = 2                                                 #Dimension of location (2D MNIST images)
    num_glimpses = 6                                            #Number of glimpses allowed
    num_classes = 10                                            #Number of classes for classification task (10 digits for MNIST)
    max_grad_norm = 0.5                                         #Hyperparameter for gradient clipping

    step = 100000                                               #Total training steps
    lr_start = 1e-3                                             #Initial learning rate
    lr_min = 1e-4                                               #Lowest learning rate when decaying

    # Monte Carlo sampling
    M = 10                                                      #How many samples we use at each iteration for REINFORCE