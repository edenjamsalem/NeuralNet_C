#include "../includes/header.hpp"

int main(void) {
	std::vector<double> inputs;
	std::vector<size_t> layer_sizes = {16, 16, 10}; // excludes input layer

	Network network(inputs, {16, 16, 10});
}