#include "../includes/header.hpp"

/*
	struct MNIST_dataset {
		OuterContainer<InnerContainer<PixelType>> training_images;
		OuterContainer<InnerContainer<PixelType>> test_images;
		OuterContainer<LabelType> training_labels;
		OuterContainer<LabelType> test_labels;
	};
*/

int main(void) {
	auto dataset = mnist::read_dataset<std::vector, std::vector, double, uint8_t>();
	
	// normalize grayscale 0.0-255.0 --> 0.0-1.0
	for (auto &image : dataset.training_images) {
		for (auto &pixel : image) {
			pixel /= 255.0;
		}
	}
	
	// create network
	size_t input_size = dataset.test_images[0].size();
	Network network({input_size, 16, 16, 10});

	// train network on each image in the training set
	for (auto &image : dataset.training_images) {
		network.trainOn(image);
	}
}