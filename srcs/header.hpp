#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

struct Neuron {
	std::vector<double> weights;
	int bias;
	double output; // ??

	// Constructor
	Neuron(int num_inputs = 0, int bias = 0.0) : weights(num_inputs), bias(bias) {}
};

typedef std::vector<Neuron> Layer;
typedef std::vector<Layer> Network;