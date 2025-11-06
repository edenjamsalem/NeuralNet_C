#pragma once
#include <vector>

template <typename T>
class flat2DArray {
	public:
		std::vector<T> data;
		size_t rows, cols;

		// Constructor
		flat2DArray(size_t r, size_t c) :  data(r*c), rows(r), cols(c) {}
		flat2DArray() :  data(0), rows(0), cols(0) {}

		// index operator
		T &operator()(size_t i, size_t j) {
			return data[(i * cols) + j];
		}
		const T &operator()(size_t i, size_t j) const {
			return data[(i * cols) + j];
		}

		void init(size_t r, size_t c) {
			data.reserve(r * c);
			rows = r;
			cols = c;
		}
};

template <typename T>
class flat3DArray {
	public:
		std::vector<T> data;
		size_t rows, cols, depths;

		// Constructor
		flat3DArray(size_t r, size_t c, size_t d) :  data(r*c*d), rows(r), cols(c), depths(d) {}
		flat3DArray() :  data(0), rows(0), cols(0), depths(0) {}

		// index operator
		T &operator()(size_t i, size_t j, size_t k) {
			return data[(i * cols * depths) + (j * depths) + k];
		}
		const T &operator()(size_t i, size_t j, size_t k) const {
			return data[(i * cols * depths) + (j * depths) + k];
		}

		void init(size_t r, size_t c, size_t d) {
			data.reserve(r * c * d);
			rows = r;
			cols = c;
			depths = d;
		}
};

template <typename T>
struct flatNDArray {
	std::vector<T> data;
	std::vector<size_t> dims;

	// Vector Constructor
	flatNDArray(std::vector<size_t> &dimensions) : dims(dimensions) {
		size_t size = 1;
		for (auto dim : dims) { size *= dim; }
		data.resize(size);
	}
	
	// Variadic Constructor
	template <typename... Sizes>
    flatNDArray(Sizes... sizes) : dims({static_cast<size_t>(sizes)...}) {
        size_t total = 1;
        for (auto d : dims) total *= d;
        data.resize(total);
    }
	
	size_t idx(const std::vector<size_t>& indices) const {
        size_t flat = 0;
        size_t multiplier = 1;
        for (int i = dims.size() - 1; i >= 0; --i) {
            flat += indices[i] * multiplier;
            multiplier *= dims[i];
        }
        return flat;
    }

	// index operator
    T& operator()(const std::vector<size_t>& indices) {
        return data[idx(indices)];
    }
    const T& operator()(const std::vector<size_t>& indices) const {
        return data[idx(indices)];
    }
};
