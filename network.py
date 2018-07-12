"""Class that represents the network to be evolved."""
import random
import logging
#from train import train_and_score

LUT = {('2','3','0.001'):0.6296,
('4','3','0.001'):0.9386,
('6','3','0.001'):0.972,
('2','2','0.001'):0.7033,
('4','2','0.001'):0.9397,
('6','2','0.001'):0.9708,
('2','3','0.0015'):0.6855,
('4','3','0.0015'):0.9542,
('6','3','0.0015'):0.973,
('2','2','0.0015'):0.749,
('4','2','0.0015'):0.9457,
('6','2','0.0015'):0.9716,
('2','3','0.002'):0.6954,
('4','3','0.002'):0.9569,
('6','3','0.002'):0.9794,
('2','2','0.002'):0.7837,
('4','2','0.002'):0.9443,
('6','2','0.002'):0.9689,
('2','3','0.003'):0.9567,
('4','3','0.003'):0.9583,
('6','3','0.003'):0.9816,
('2','2','0.003'):0.7923,
('4','2','0.003'):0.9471,
('6','2','0.003'):0.9651
};
"""
my_dict = dict.fromkeys(['a', 'b', 'c'], 10)
my_dict.update(dict.fromkeys(['b', 'e'], 20))
"""
class Network():
    """Represent a network and let us operate on it.

    Currently only works for an MLP.
    """

    def __init__(self, nn_param_choices=None):
        """Initialize our network.
        """
        self.accuracy = 0.
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): represents MLP network parameters

    def create_random(self):
        """Create a random network."""
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network):
        """Set network properties.

        Args:
            network (dict): The network parameters

        """
        self.network = network

    def train(self, dataset):
        """Train the network and record the accuracy.

        Args:
            dataset (str): Name of dataset to use.

        """
		# hard code all accuracy

        if self.accuracy == 0.:
            self.accuracy = LUT[str(self.network['features']),
			str(self.network['nb_layers']),str(self.network['learning_rate'])]

    def print_network(self):
        """Print out a network."""
        print(self.network)
        print("Network accuracy: %.2f%%" % (self.accuracy * 100))
