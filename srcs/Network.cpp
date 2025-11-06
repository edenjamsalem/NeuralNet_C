#include "../includes/Network.hpp"

Network::Network(const std::vector<size_t> &layer_sizes) : 
	num_layers(layer_sizes.size() - 1), 
	max_layer_len(*std::max_element(layer_sizes.begin() + 1, layer_sizes.end())),
	layer_sizes(layer_sizes),

	input_layer(layer_sizes[0]), 
	network(num_layers, max_layer_len)
{
	// init weights
	for (size_t layer = 0; layer < this->num_layers; layer++) {
		this->weights[layer].init(layer_sizes[layer + 1], layer_sizes[layer]);

		// seed random values
		for (size_t to = 0; to < this->layer_sizes[layer + 1]; to++) { 	// indexes backwards
			for (size_t from = 0; from < this->layer_sizes[layer]; from++) {
				this->weights[layer](to, from) = gen_random_double();
			}
		}
	}
}

void Network::setInputs(std::vector<float> image) {
	this->input_layer = std::move(image);
}

void Network::feedForward() {
	// handle rest of network
	for (size_t layer = 0; layer < num_layers; layer++) {
		for (size_t to = 0; to < this->layer_sizes[layer + 1]; to++) {
			float z = 0;
			for (size_t from = 0; from < this->layer_sizes[layer]; from++) {
				// network[0] is first hidden layer so [layer - 1] is needed
				float signal = (layer == 0) ? this->input_layer[from] : this->network(layer - 1, from).signal;
				z += this->weights[layer](to, from) * signal;
			}
			float bias = this->network(layer, to).bias;
			this->network(layer, to).signal = sigmoid(z + bias);
		}
	}
}

void Network::trainOn(std::vector<float> image, uint8_t expected_ouput) {
	(void)expected_ouput;
	this->setInputs(image);
	this->feedForward();
	// this->backProp();
}

void Network::backProp() {

}
