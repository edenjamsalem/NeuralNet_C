#include "../include/Network.hpp"

/* 
	--- Constructors --- 
*/

NeuralNetwork::NeuralNetwork(const std::vector<size_t> &layout) : 
	inputActivations(layout.front()),
	batchScale(η / miniBatchSize)
{
	// validate network size
	if (layout.size() < 3) {
		throw std::invalid_argument("Network must have at least 1 hidden layer.");
	}

	// allocate network buffer
	network.reserve(layout.size() - 1);

	for (size_t layer = 0; layer < layout.size() - 1; ++layer) {
		size_t rows = layout[layer + 1];
		size_t cols = layout[layer];
		this->buffer_size += (2 * rows * cols) + (4 * rows);
	}
	buffer = std::make_unique<float[]>(this->buffer_size);

	// assign map views of each layer
	float *start = this->buffer.get();
	for (size_t layer = 0; layer < layout.size() - 1; ++layer) {
		size_t rows = layout[layer + 1];
		size_t cols = layout[layer];

		network.emplace_back(LayerView(start, rows, cols));
		start += (2 * rows * cols) + (4 * rows);
	}
}

NeuralNetwork::NeuralNetwork(std::vector<size_t> &layer_sizes) 
	: NeuralNetwork(static_cast<const std::vector<size_t>&>(layer_sizes)) 
{}

/* 
	--- Public Methods --- 
*/

// Uses mini-batch 'Stochastic Gradient Descent' to train model on the training_data
void NeuralNetwork::trainModelSGD(std::vector<std::vector<float>> &training_data, std::vector<uint8_t> &training_labels) {
	Eigen::VectorXf expectedOutput(this->network.back().activations.size());

	// Used to track network's improvement across batches
	size_t progressBar = 1;
	size_t progressBarMax = 40;

	// train network on each image in the dataset
	for (size_t i = 0; i < training_data.size(); ++i) {
		this->_feedForward(training_data[i]);

		// vectorize exepected output 
		expectedOutput.setZero();
		expectedOutput[training_labels[i]] = 1.0f;

		this->_backProp(expectedOutput);

		// adjust parameters after each mini-batch
		if ((i > 1 && i % this->miniBatchSize == 0) || i == training_data.size() - 1) {
			this->_adjustNetwork();

			// update progress bar
			if (i >= (progressBar * training_data.size() / progressBarMax)) {
				std::cerr << "|";
				std::cerr.flush();
				progressBar++;
			}
		}
	}
	std::cerr << std::endl;
}

float NeuralNetwork::test(std::vector<std::vector<float>> &test_data, std::vector<uint8_t> &test_labels) {
	size_t successCount = 0;
	Eigen::Index prediction;

	for (size_t i = 0; i < test_data.size(); i++) {
		// feed each image into the network
		this->_feedForward(test_data[i]);

		// compare predicted output to expected value
		this->network.back().activations.maxCoeff(&prediction);
		successCount += (static_cast<uint8_t>(prediction) == test_labels[i]);

		// std::cerr << "Prediction: " << prediction << "\n";
		// std::cerr << "Expected: " << static_cast<int>(test_labels[i]) << "\n";
	}
	return (static_cast<float>(successCount) / test_data.size());
}

// Save weights and biases to binary file
void NeuralNetwork::saveModel(const std::string &filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open file for saving model");

    for (auto &layer : network) {
        out.write(reinterpret_cast<char*>(layer.weights.data()), layer.weights.size() * sizeof(float));
        out.write(reinterpret_cast<char*>(layer.biases.data()), layer.biases.size() * sizeof(float));
    }
}

// Load weights and biases from binary file
void NeuralNetwork::loadModel(const std::string &filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open file for loading model");

    for (auto &layer : network) {
        in.read(reinterpret_cast<char*>(layer.weights.data()), layer.weights.size() * sizeof(float));
        in.read(reinterpret_cast<char*>(layer.biases.data()), layer.biases.size() * sizeof(float));
        if (!in) throw std::runtime_error("Error reading model from file");
    }
}

// Save weights and biases to binary file
void NeuralNetwork::saveModel(std::ostream &out) {
    if (!out) throw std::runtime_error("Cannot open file for saving model");

    for (auto &layer : network) {
        out.write(reinterpret_cast<char*>(layer.weights.data()), layer.weights.size() * sizeof(float));
        out.write(reinterpret_cast<char*>(layer.biases.data()), layer.biases.size() * sizeof(float));
    }
}

// Load weights and biases from binary file
void NeuralNetwork::loadModel(std::istream &in) {
    if (!in) throw std::runtime_error("Cannot open file for loading model");

    for (auto &layer : network) {
        in.read(reinterpret_cast<char*>(layer.weights.data()), layer.weights.size() * sizeof(float));
        in.read(reinterpret_cast<char*>(layer.biases.data()), layer.biases.size() * sizeof(float));
        if (!in) throw std::runtime_error("Error reading model from file");
    }
}

/* 
	--- Private Methods --- 
*/

void NeuralNetwork::_feedForward(const std::vector<float> &image) {
	// copy inputs so original image is not overwritten
	this->inputActivations = Eigen::VectorXf(Eigen::Map<const Eigen::VectorXf>(image.data(), image.size()));
	Eigen::VectorXf activations = this->inputActivations;
    
	for (auto &layer : this->network) {
        layer.activations.noalias() = (layer.weights * activations) + layer.biases;
		layer.activations.noalias() = (&layer == &this->network.back()) ? softmax(layer.activations) : layer.activations.unaryExpr(&ReLU);
        activations = layer.activations;
    }
}

void NeuralNetwork::_backProp(Eigen::VectorXf &expectedOutput) {
	// Derivation of δL 		: how much each output node contributed to the final cost
	// C = 1/2∑j(yj−aLj) 		: cost function
	// ∇aC = ∂C/∂aL = aL−y  	: rate of change of cost with respect to output activations aL
    // δL = (aL−y) ⊙ σ′(zL) 	: aL replaces zL here to minimise memory use, sigmoidPrime is updated accordingly
	
	this->network.back().delta = this->network.back().activations - expectedOutput;

	// Backpropagate delta
	for (int layer = static_cast<int>(this->network.size()) - 2; layer >= 0; --layer) {
		LayerView &current = this->network[layer];
		LayerView &next = this->network[layer + 1];

		//δl = ((wl+1)Tδl+1) ⊙ σ′(zl),
		current.delta = (next.weights.transpose() * next.delta).cwiseProduct(current.activations.unaryExpr(&ReLUPrime));
	}

	// Compute Gradients
	Eigen::VectorXf prevActivations = this->inputActivations;
	for (auto &layer : this->network) {
		layer.dW += layer.delta * prevActivations.transpose();
		layer.db += layer.delta;
		prevActivations = layer.activations;
	}
}

void NeuralNetwork::_adjustNetwork() {
	for (auto &layer : this->network) {
		// apply changes to weights and biases 
		layer.weights -= this->batchScale * layer.dW;
		layer.biases -= this->batchScale * layer.db;
		
		// reset for next batch 
		layer.dW.setZero();
		layer.db.setZero();
		layer.delta.setZero();
    }
}
