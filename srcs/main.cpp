#include "../includes/header.hpp"

/*
	struct MNIST_dataset {
		OuterContainer<InnerContainer<PixelType>> training_images;
		OuterContainer<InnerContainer<PixelType>> test_images;
		OuterContainer<LabelType> training_labels;
		OuterContainer<LabelType> test_labels;
	};
*/

int main() {
	auto dataset = mnist::read_dataset<std::vector, std::vector, float, uint8_t>("./includes/mnist");
	std::cout << "Data imported!\n";
	
	// grayscale 0.0-255.0 --> 0.0-1.0
	mnist::normalize_dataset(dataset);
	
	// create network
	size_t input_size = dataset.test_images[0].size(); // assumes inputs size is the same for all images
	NeuralNetwork network({input_size, 16, 16, 10});
	std::cout << "Network created!\n";

	// Apply Mini-batch Stochastic Gradient Descent algorithm to training data
	network.SGD(dataset);
	std::cout << "Finished training set of " << dataset.training_images.size() << " images!\n";
}