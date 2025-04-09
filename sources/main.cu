#include "perceptron.hxx"
#include <iostream>

int main() {
	neural_networks::perceptron nn = { 4, 4, 2 };
	gpu_math::vector v(4, 3.14f);
	std::cout << nn.activate(v);
	nn.dump("weights.dump");
	return 0;
}
