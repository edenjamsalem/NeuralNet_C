#include "../includes/header.hpp"

Network::Network(const std::vector<size_t> &layer_sizes) : 
	num_layers(layer_sizes.size() - 1), 
	max_layer_len(*std::max_element(layer_sizes.begin() + 1, layer_sizes.end())),
	layer_sizes(layer_sizes),

	input_layer(layer_sizes[0]),
	network(num_layers)
{
	for (size_t layer = 0; layer < this->num_layers; layer++) {
		size_t rows = layer_sizes[layer + 1];
		size_t cols = layer_sizes[layer];
		
		this->network[layer].biases.resize(rows);
		this->network[layer].activations.resize(rows);
		this->network[layer].weights.resize(rows, cols);

		this->network[layer].biases.setRandom();
		this->network[layer].activations.setZero();
		this->network[layer].weights.setRandom();
	}
}

void Network::setInputs(std::vector<float> image) {
	this->input_layer = Eigen::VectorXf::Map(image.data(), image.size());
}

void Network::SGD(mnist::MNIST_dataset<std::__1::vector, std::__1::vector<float, std::__1::allocator<float>>, uint8_t> dataset) {
	// vectorize expected output
	Eigen::VectorXf expected_output(this->layer_sizes.back());

	// train network on each image in the dataset
	for (size_t i = 0; i < dataset.training_images.size(); i++) {
		expected_output.setZero();
		expected_output[dataset.training_labels[i]] = 1.0f;

		this->trainOn(dataset.training_images[i], expected_output);

		// adjust parameters after each mini-batch
		if ((i > 1 && i % MINI_BATCH_SIZE == 0)) {
			// this->adjust_parameters();
		}
	}
}

void Network::trainOn(std::vector<float> image, Eigen::VectorXf expected_ouput) {
	(void)expected_ouput;

	this->setInputs(image);
	this->feedForward();
	size_t cost = this->calculateCost();
	(void)cost;
	// this->backProp();
}

void Network::feedForward() {
	Eigen::VectorXf a = this->input_layer; // a ==> current layer activations

    for (size_t layer = 0; layer < this->num_layers; ++layer) {
        Eigen::VectorXf z = (this->network[layer].weights * a) + this->network[layer].biases;
        a = z.unaryExpr(&sigmoid);
        this->network[layer].activations = a;
    }	
}

size_t Network::calculateCost() {
	// for (size_t i = 0;)
	return (0);
}

void Network::backProp() {

}
