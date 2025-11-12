#include "NeuralNetwork/include/Network.hpp"
#include "mnist/include/mnist/mnist_reader.hpp"
#include "mnist/include/mnist/mnist_utils.hpp"

int main(int argc, char **argv) {
	std::cout << "Loading Mnist Data!\n";
	auto dataset = mnist::read_dataset<std::vector, std::vector, float, uint8_t>("./mnist");
	mnist::normalize_dataset(dataset); // grayscale 0.0-255.0 --> 0.0-1.0
	
	// Create network
	std::cout << "Creating Neural Network!\n";
	NeuralNetwork network({dataset.test_images[0].size(), 128, 64, 10}); // assumes inputs size is the same for all images
	const std::string filename("NN.bin");

	if (argc < 2) {
		// Train model using Mini-batch Stochastic Gradient Descent algorithm
		std::cout << "Training model...\n";
		network.trainModelSGD(dataset.training_images, dataset.training_labels);
		std::cout << "Finished training set of " << dataset.training_images.size() << " images!\n";

		// Save model to binary file
		std::cout << "Model saved to " << filename << "!\n";
		network.saveModel(filename);
	}
	else {
		// Load model from binary file
		std::cout << "Loading model from " << argv[1] << "!\n";
		network.loadModel(filename);
	}

	// Run model with test data to find rate of successful predictions
	std::cout << "Running test data!\n";
	float successRate = network.test(dataset.test_images, dataset.test_labels);
	std::cout << "Model predicted test data with " << (successRate * 100) << "% accuracy!\n";

}