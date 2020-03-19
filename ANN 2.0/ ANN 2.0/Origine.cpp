#include <iostream>
#include <vector>
#include <math.h>

double Sigmoid(const double& f)
{
	return 1 / (1 + exp(-f));
}

double SigmoidDerivative(const double& f)
{
	return (Sigmoid(f) * (1 - Sigmoid(f)));
}

double ReLu(const double& f)
{
	return 1 / (1 + exp(-f));
}

double ReLuDerivative(const double& f)
{
	return (Sigmoid(f) * (1 - Sigmoid(f)));
}

class Neuron {
public:
	Neuron();
	Neuron(std::vector<Neuron*> *_prev_neurons, std::vector<Neuron*> *_next_neurons);
	void SetPrevNeurons(std::vector<Neuron*> *_prev_neurons);
	void SetNextNeurons(std::vector<Neuron*> *_next_neurons);
	double ComputeValue();
	void BackPropagation(double learning_rate);
	void setFunctions(double (*_function)(const double&), double (*_function_derivative)(const double&));

	std::vector<Neuron*> *next_neurons;
	std::vector<Neuron*> *prev_neurons;
	std::vector<double> weights;
	double z;
	double bias;
	double error;
	double output;
	double (*function)(const double&);
	double (*function_derivative)(const double&);
};

Neuron::Neuron()
{
	this->bias = 0;
	this->z = 0;
	this->error = 0;
}

Neuron::Neuron(std::vector<Neuron*> *_prev_neurons, std::vector<Neuron*> *_next_neurons)
{
	this->bias = 0;
	this->z = 0;
	this->error = 0;
	this->SetNextNeurons(_next_neurons);
	this->SetPrevNeurons(_prev_neurons);
}


void Neuron::SetPrevNeurons(std::vector<Neuron*> *_prev_neurons)
{
	this->prev_neurons = _prev_neurons;
	this->weights.resize(this->prev_neurons->size());
	for (unsigned int i = 0; i < this->weights.size(); i++)
	{
		this->weights[i] = (rand() % (100+1))/(this->prev_neurons->size()*2);
	}
}

void Neuron::SetNextNeurons(std::vector<Neuron*> *_next_neurons)
{
	this->next_neurons = _next_neurons;
}

double Neuron::ComputeValue()
{
	this->z = 0;
	for (unsigned int i = 0; i < (*this->prev_neurons).size(); i++) {
		this->z += ((*prev_neurons)[i]->output) * this->weights[i];
	}
	this->z += this->bias;
	this->output = this->function(this->z);

	return this->output;
}

void Neuron::BackPropagation(double learning_rate)
{
	for (unsigned int i = 0; i < (*this->prev_neurons).size(); i++) {
		(*this->prev_neurons)[i]->error = this->weights[i] * this->error * this->function_derivative(this->z);
		this->weights[i] += (*this->prev_neurons)[i]->error * learning_rate;
	}

	this->bias += this->error * learning_rate;

}

void Neuron::setFunctions(double(*_function)(const double&), double(*_function_derivative)(const double&))
{
	this->function = _function;
	this->function_derivative = _function_derivative;
}


typedef std::vector<Neuron*> Layer;


class NeuralNetwork {
public:
	void SetInputSize(int size);
	void AddLayer(int size, double(*_function)(const double&), double(*_function_derivative)(const double&));
	void LinkLayers();
	std::vector<double> ComputeOutput(std::vector<double> in);
	double Learn(std::vector<double> input, std::vector<double> expected_output, double learning_rate);
private:
	Layer* input_layer;
	std::vector<Layer*> hiden_layers;
};


void NeuralNetwork::SetInputSize(int size)
{
	this->input_layer = new Layer;
	for (int i = 0; i < size; i++)
	{
		this->input_layer->push_back(new Neuron);
	}
}

void NeuralNetwork::AddLayer(int size, double(*_function)(const double&), double(*_function_derivative)(const double&))
{
	Layer* layer = new Layer;
	for (int i = 0; i < size; i++)
	{
		Neuron* neuron = new Neuron;
		neuron->setFunctions(_function, _function_derivative);
		layer->push_back(neuron);
	}
	this->hiden_layers.push_back(layer);
}

void NeuralNetwork::LinkLayers()
{
	Layer* prev_layer_ptr = this->input_layer;

	for (unsigned int i = 0; i < this->hiden_layers.size(); i++)
	{
		for (unsigned int e = 0; e < (*prev_layer_ptr).size(); e++)
		{
			(*prev_layer_ptr)[e]->SetNextNeurons(this->hiden_layers[i]);
		}


		for (unsigned int e = 0; e < this->hiden_layers[i]->size(); e++)
		{
			(*this->hiden_layers[i])[e]->SetPrevNeurons(prev_layer_ptr);
		}
		prev_layer_ptr = this->hiden_layers[i];
	}
}

std::vector<double> NeuralNetwork::ComputeOutput(std::vector<double> in)
{
	if ((*this->input_layer).size() != in.size()) {
		std::cout << "Invalid input vector size in NeuralNetwork::ComputeOutput\n";
		exit(-1);
	}

	for (unsigned int i = 0; i < in.size(); i++) {
		(*this->input_layer)[i]->output = in[i];
	}

	for (unsigned int i = 0; i < this->hiden_layers.size(); i++)
	{
		for (unsigned int e = 0; e < this->hiden_layers[i]->size(); e++)
		{
			(*(this->hiden_layers[i]))[e]->ComputeValue();
		}
	}

	std::vector<double> out;

	Layer* last_layer = this->hiden_layers[this->hiden_layers.size() - 1];

	for (unsigned int i = 0; i < last_layer->size(); i++)
	{
		out.push_back((*last_layer)[i]->output);
	}

	return out;
}

double NeuralNetwork::Learn(std::vector<double> input, std::vector<double> expected_output, double learning_rate)
{
	std::vector<double> output = this->ComputeOutput(input);
	std::vector<double> error(expected_output.size());

	for (unsigned int i = 0; i < error.size(); i++)
	{
		error[i] = (pow(expected_output[i] - output[i], 2)) / 2;
	}

	Layer* last_layer = this->hiden_layers[this->hiden_layers.size() - 1];
	for (unsigned int i = 0; i < last_layer->size(); i++)
	{
		(*last_layer)[i]->error = error[i];
	}

	for (unsigned int i = 0; i < this->hiden_layers.size(); i++) {
		for (unsigned int e = 0; e < this->hiden_layers[i]->size(); e++) {
			(*this->hiden_layers[i])[e]->BackPropagation(learning_rate);
		}
	}

	double scalar_error = 0;
	for (unsigned int i = 0; i < error.size(); i++)
	{
		scalar_error += error[i];
	}
	
	scalar_error /= error.size();

	return scalar_error;
}

int main()
{
	//create neural network
	NeuralNetwork ann;
	ann.SetInputSize(2);
	ann.AddLayer(32, ReLu, ReLuDerivative);
	ann.AddLayer(32, ReLu, ReLuDerivative); 
	ann.AddLayer(32, ReLu, ReLuDerivative);
	ann.AddLayer(1, ReLu, ReLuDerivative);
	ann.LinkLayers();

	//Vector of two elements
	std::vector<double> input;
	input.push_back(1.f);
	input.push_back(1.f);

	//get the output
	std::vector<double> output = ann.ComputeOutput(input);

	for (unsigned int i = 0; i < output.size(); i++)
	{
		std::cout << "[" << i << "]: " << output[i] << std::endl;
	}
}
