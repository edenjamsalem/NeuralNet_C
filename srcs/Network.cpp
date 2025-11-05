#include "../includes/Network.hpp"

Network::Network(const std::vector<size_t> &layer_sizes) : 
	num_layers(layer_sizes.size()), 
	max_layer_len(*std::max_element(layer_sizes.begin() + 1, layer_sizes.end())),
	layer_sizes(layer_sizes),

	input_layer(layer_sizes[0]), 
	network(num_layers, max_layer_len)
{
	// init weights
	for (size_t layer = 0; layer < this->num_layers - 1; layer++) {
		weights[layer].init(layer_sizes[layer], layer_sizes[layer + 1]);

		// seed random values
		for (size_t from = 0; from < this->layer_sizes[layer]; from++) {
			for (size_t to = 0; to < this->layer_sizes[layer + 1]; to++) {
				this->weights[layer](from, to) = gen_random_double();
			}
		}
	}
}

void Network::setInputs(std::vector<double> image) {
	this->input_layer = std::move(image);
}

void Network::feedForward() {
	// handle input layer
	for (size_t to = 0; to < this->layer_sizes[1]; to++) {
		double z = 0;
		for (size_t from = 0; from < this->layer_sizes[0]; from++) {
			double signal = this->input_layer[from];
			z += this->weights[0](from, to) * signal;
		}
		double bias = this->network(0, to).bias;
		this->network(0, to).signal = sigmoid(z + bias);
	}

	// handle rest of network
	for (size_t layer = 1; layer < num_layers - 1; layer++) {
		for (size_t to = 0; to < this->layer_sizes[layer + 1]; to++) {
			double z = 0;
			for (size_t from = 0; from < this->layer_sizes[layer]; from++) {
				z += this->weights[layer](from, to) * this->network(layer, from).signal;
			}
			double bias = this->network(layer, to).bias;
			this->network(layer, to).signal = sigmoid(z + bias);
		}
	}
}

void Network::trainOn(std::vector<double> image) {
	this->setInputs(image);
	this->feedForward();
	// this->backProp();
}
