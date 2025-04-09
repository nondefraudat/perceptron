#include "perceptron.hxx"
#include <execution>
#include <sstream>
#include <fstream>
#include <cmath>

namespace neural_networks {

__global__ void activate(float data[], uint16_t size);

perceptron::perceptron(size_t layer_count) : _weights_count(layer_count - 1) {
	_biases = std::shared_ptr<gpu_math::vector[]>(
			new gpu_math::vector[_weights_count]);
	_weights = std::shared_ptr<gpu_math::matrix[]>(
			new gpu_math::matrix[_weights_count]);
}

perceptron::perceptron(std::initializer_list<uint16_t> layer_sizes)
		: perceptron(layer_sizes.size()) {
	if (layer_sizes.size() < 1) {
		throw std::invalid_argument("weights count must be more then 0");
	}
	const uint16_t *current_size = layer_sizes.begin();
	gpu_math::matrix *current_weights = _weights.get();
	gpu_math::vector *current_bias = _biases.get();
	uint16_t width = *current_size++;
	for (; current_size != layer_sizes.end();
			current_size++, current_weights++, current_bias++) {
		uint16_t height = *current_size;
		*current_weights = gpu_math::matrix(height, width, 1.f);
		*current_bias = gpu_math::vector(height, 0.f);
		width = height;
	}
}

perceptron perceptron::load(std::string dump_name) {
	std::stringstream ss;
	ss << std::ifstream(dump_name).rdbuf();
	size_t weights_count;
	ss >> weights_count;
	perceptron p(weights_count);
	for (size_t i = 0; i < weights_count; i++) {
		uint16_t height, width;
		ss >> height >> width;
		gpu_math::matrix &matrix = p._weights[i] =
				gpu_math::matrix(height, width);
		float value;
		for (uint16_t row = 0; row < height; row++) {
			for (uint16_t column = 0; column < width; column++) {
				ss >> value;
				matrix.set(row, column, value);
			}
		}
		gpu_math::vector &vector = p._biases[i] =
				gpu_math::vector(height);
		for (uint16_t index = 0; index < height; index++) {
			ss >> value;
			vector.set(index, value);
		}
	}
	return p;
}

void perceptron::dump(std::string dump_name) noexcept
{
	std::stringstream ss;
	ss << _weights_count << '\n';
	for (size_t i = 0; i < _weights_count; i++) {
		auto &weights = _weights[i];
		ss << weights.height() << ' ' << weights.width() << '\n';
		for (uint16_t row = 0; row < weights.height(); row++) {
			ss << weights.get(row, 0);
			for (uint16_t column = 1; column < weights.width(); column++) {
				ss << ' ' << weights.get(row, column);
			}
			ss << '\n';
		}
		auto &bias = _biases[i];
		ss << bias.get(0);
		for (uint16_t index = 1; index < weights.height(); index++) {
			ss << ' ' << bias.get(index);
		}
		ss << '\n';
	}
	std::ofstream(dump_name) << ss.str();
}

gpu_math::matrix perceptron::activate(const gpu_math::vector &signals) {
	gpu_math::matrix buffer, result = signals;
	for (size_t i = 0; i < _weights_count; i++) {
		buffer = _weights[i]*result + _biases[i];
		std::swap(buffer, result);
		neural_networks::activate<<<result.size(), 1>>>(
				result.device_data(), result.size());
	}
	return result;
}

__global__ void activate(float data[], uint16_t size) {
	auto index = blockIdx.x;
	if (index < size) {
		auto &value = data[index];
		value = 1/(1 + std::exp(-value));
	}
}

}