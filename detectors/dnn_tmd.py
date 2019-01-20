from detectors.nn_tmd import NeuralNetworkTMD


class DeepNeuralNetworkTMD(NeuralNetworkTMD):
    """
    Wrapper that uses TensorFlow to allow training and
    using a deep feedforward neural network for
    travel mode detection through smartphone sensors
    """

    DEFAULT_MODEL_PATH = 'dnn_tmd'
    MINIMUM_HIDDEN_LAYERS = 2
    DEFAULT_HIDDEN_LAYERS = 2
    MAXIMUM_HIDDEN_LAYERS = None
    DEFAULT_OPTIMIZER = 'adagrad'
    DEFAULT_HIDDEN_ACTIVATION = 'leaky_relu'

    def __init__(self, save_path=DEFAULT_MODEL_PATH, **kwargs):
        super(DeepNeuralNetworkTMD, self).__init__(save_path=save_path, **kwargs)

    def fit(
            self,
            n_hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
            optimizer: str = DEFAULT_OPTIMIZER,
            **kwargs
    ):
        """
        Pre process input and train a Deep Neural Network for travel mode detection
        using mini-batch Backpropagation with l2 regularization and dropout
        :param n_hidden_layers: Number of hidden layers in the neural network
        :param optimizer: Optimization Algorithm used to train model ('adam', 'adagrad', 'momentum', 'gradient_descent')
        :param kwargs: additional arguments
        :return:
        """
        super(DeepNeuralNetworkTMD, self).fit(
            n_hidden_layers=n_hidden_layers,
            optimizer=optimizer,
            **kwargs
        )
