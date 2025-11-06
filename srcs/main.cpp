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
	auto dataset = mnist::read_dataset<std::vector, std::vector, double, uint8_t>("./includes/mnist");
	std::cout << "Data imported!\n";
	
	// normalize grayscale 0.0-255.0 --> 0.0-1.0
	for (auto &image : dataset.training_images) {
		for (auto &pixel : image) {
			pixel /= 255.0;
		}
	}
	
	// create network
	size_t input_size = dataset.test_images[0].size(); // assumes inputs size is the same for all images
	Network network({input_size, 16, 16, 10});
	std::cout << "Network created!\n";

	// train network on each image in the training set
	for (auto &image : dataset.training_images) {
		network.trainOn(image);
	}
	std::cout << "Finished training set!\n";
}