#include "../includes/Network.hpp"

Network::Network(std::vector<double> &inputs, const std::vector<size_t> &layer_sizes) : 
	num_layers(layer_sizes.size()), // excludes input layer
	max_len(*std::max_element(layer_sizes.begin(), layer_sizes.end())),
	
	input_layer(inputs), 
	input_weights(layer_sizes[0], inputs.size()),
	
	network(num_layers, max_len),
	weights(num_layers - 1, max_len, max_len)
{
	// seed random values for weights 
	for (size_t i = 0; i < input_weights.rows; i++) {
		for (size_t j = 0; j < input_weights.cols; j++) {
			input_weights(i, j) = gen_random_double();
		}
	}
	for (size_t i = 1; i < num_layers; i++) {
		for (size_t j = 0; j < layer_sizes[i]; j++) {
			for (size_t k = 0; k < layer_sizes[i - 1]; k++) {
				weights(i - 1, j, k) = gen_random_double();
			}
		}
	}
}