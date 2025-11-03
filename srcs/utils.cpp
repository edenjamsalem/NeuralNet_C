#include <random>

double gen_random_double() {
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<double> dist(-1.0, 1.0);
    return dist(gen);
}