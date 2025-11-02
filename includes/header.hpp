#include <vector>
#include <random>

#define NUM_LAYERS 3			// does not include input layer
#define NEURONS_PER_LAYER 16	// does not include input layer

double gen_random_double() {
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<double> dist(-1.0, 1.0);
    return dist(gen);
}

struct flat2DArray {
	std::vector<double> data;
	size_t rows, cols;

	// Constructor
	flat2DArray(size_t r, size_t c) :  data(r*c), rows(r), cols(c) {}
	// flat2DArray() : rows(0), cols(0), data(0) {} // ??

	// index operator
	double &operator()(size_t i, size_t j) {
		return data[(i * cols) + j];
	}
};

struct Neuron {
	double bias;
	double signal;

	// Constructor
    Neuron() : bias(gen_random_double()), signal(0.0) {}
};

struct Network {
	// input values and weights heap allocated as they are potentially large and variable
	std::vector<double> input_layer;
	flat2DArray input_weights;
	
	// rest of network is stack allocated
	Neuron network[NUM_LAYERS][NEURONS_PER_LAYER];
	double weights[NUM_LAYERS - 1][NEURONS_PER_LAYER][NEURONS_PER_LAYER];

	Network(std::vector<double> &inputs);
	// ~Network();

};

Network::Network(std::vector<double> &inputs) : input_layer(inputs), input_weights(NEURONS_PER_LAYER, inputs.size()) {
	// seed random values for weights 
	for (size_t i = 0; i < input_weights.rows; i++) {
		for (size_t j = 0; j < input_weights.cols; j++) {
			input_weights(i, j) = gen_random_double();
		}
	}
	for (size_t i = 0; i < NUM_LAYERS - 1; i++) {
		for (size_t j = 0; j < NEURONS_PER_LAYER; j++) {
			for (size_t k = 0; k < NEURONS_PER_LAYER; k++) {
				weights[i][j][k] = gen_random_double();
			}
		}
	}
}
