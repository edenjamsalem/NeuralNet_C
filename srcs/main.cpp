#include "header.hpp"

Network create_network(std::vector<int> neurons_per_layer) {
	Network network;
	network.reserve(neurons_per_layer.size());

	for (size_t i = 0; i < neurons_per_layer.size(); i++) {
		Layer current_layer = Layer();
		current_layer.reserve(neurons_per_layer[i]);

		for (int j = 0; j < neurons_per_layer[i]; j++) {
			int num_inputs = (i > 0) ? neurons_per_layer[i - 1] : 0;
			current_layer.emplace_back(num_inputs);
		}
		network.push_back(std::move(current_layer));
	}
	return (network);
}

int main(void) {
	std::vector<double> inputs;
	std::vector<int> neurons_per_layer = {(int)inputs.size(), 16, 16, 10};

	Network network = create_network(neurons_per_layer);
}