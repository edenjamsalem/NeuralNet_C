#include "NeuralNetwork/include/Network.hpp"
#include "mnist/include/mnist/mnist_reader.hpp"
#include "mnist/include/mnist/mnist_utils.hpp"
#include "unistd.h"

int main() {
	bool input_redirected = !isatty(fileno(stdin));		// detects '<'
	bool output_redirected = !isatty(fileno(stdout));	// detects '>'

	try {
		std::cerr << "Loading Mnist Dataset!\n";
		auto dataset = mnist::read_dataset<std::vector, std::vector, float, uint8_t>("./mnist");
		mnist::normalize_dataset(dataset); // grayscale 0.0-255.0 --> 0.0-1.0
		
		std::cerr << "Creating Neural Network!\n";
		NeuralNetwork network({dataset.test_images[0].size(), 128, 32, 10});

		if (input_redirected) {
			// Load model from binary file
			std::cerr << "Loading model from binary file\n";
			network.loadModel(std::cin);
		}
		else {
			// Train model using Mini-batch Stochastic Gradient Descent algorithm
			std::cerr << "Training new model...\n";
			network.trainModelSGD(dataset.training_images, dataset.training_labels);
			std::cerr << "Finished training set of " << dataset.training_images.size() << " images!\n";
		}
		if (output_redirected) {
			// Save model to binary file
			std::cerr << "Model saved to binary file\n";
			network.saveModel(std::cout);
		}

		// Run model with test data to find rate of successful predictions
		std::cerr << "Running test data!\n";
		float successRate = network.test(dataset.test_images, dataset.test_labels);
		std::cerr << "Model predicted test data with " << (successRate * 100) << "% accuracy!\n";
	}
	catch (std::exception &err) {
		std::cerr << "Aborted: " << err.what() << std::endl;
	}
}