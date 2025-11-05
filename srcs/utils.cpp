#include <random>

double gen_random_double() {
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<double> dist(-1.0, 1.0);
    return dist(gen);
}

double sigmoid(double x) {
    return (1.0 / (1.0 + std::exp(-x)));
}

double ReLU(double x) {
    return (x < 0.0 ? 0.0 : x);
}