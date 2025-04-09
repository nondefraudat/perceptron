#pragma once

#include <gpu-math/matrix.hxx>
#include <gpu-math/vector.hxx>
#include <initializer_list>
#include <string>

namespace neural_networks {

class perceptron {
	size_t _weights_count;
	std::shared_ptr<gpu_math::vector[]> _biases;
	std::shared_ptr<gpu_math::matrix[]> _weights;
	
public:
	perceptron(size_t layer_count);
	perceptron(std::initializer_list<uint16_t> layer_sizes);
	
	static perceptron load(std::string dump_name);
	void dump(std::string dump_name) noexcept;

	gpu_math::matrix activate(const gpu_math::vector &signals);
};

}
