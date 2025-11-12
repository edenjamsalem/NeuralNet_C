#include "NeuralNetwork/include/Network.hpp"
#include "mnist/include/mnist/mnist_reader.hpp"
#include "mnist/include/mnist/mnist_utils.hpp"

int main() {
	auto dataset = mnist::read_dataset<std::vector, std::vector, float, uint8_t>("./mnist");
	std::cout << "Data imported!\n";
	
	// grayscale 0.0-255.0 --> 0.0-1.0
	mnist::normalize_dataset(dataset);
	
	// Create network
	std::cout << "Creating Neural Network!\n";
	NeuralNetwork network({dataset.test_images[0].size(), 128, 64, 10}); // assumes inputs size is the same for all images

	// Apply Mini-batch Stochastic Gradient Descent algorithm to training data
	std::cout << "Training model...\n";
	network.trainModelSGD(dataset.training_images, dataset.training_labels);
	std::cout << "Finished training set of " << dataset.training_images.size() << " images!\n";

	// Run model with test data to find rate of successful predictions
	std::cout << "Running test data!\n";
	float successRate = network.test(dataset.test_images, dataset.test_labels);
	std::cout << "Model predicted test data with " << (successRate * 100) << "% accuracy!\n";
}