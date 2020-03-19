 # **Ann 2.0**
 *a simple implementation of a neural network in pure c++, only one file * 
 
 
 
 ## Build
 you can build it with visual studio 19, or using g++ with:
 

  

    g++ Origine.cpp

## code example

    NeuralNetwork ann;
	ann.SetInputSize(2);
	ann.AddLayer(32, ReLu, ReLuDerivative);
	ann.AddLayer(32, ReLu, ReLuDerivative); 
	ann.AddLayer(32, ReLu, ReLuDerivative);
	ann.AddLayer(1, ReLu, ReLuDerivative);
	ann.LinkLayers();
	
	//two elemets vector
	std::vector<double> input;
	input.push_back(1.f);
	input.push_back(1.f);
	
	std::vector<double> output = ann.ComputeOutput(input);

	for (unsigned int i = 0; i < output.size(); i++)
	{
		std::cout << "[" << i << "]: " << output[i] << std::endl;
	}
  
  
instantiate the object of the class `NeuralNetwork`

    NeuralNetwork ann
    
set the input size of the neural network

    ann.SetInputSize(size);
add a new layer to the neural network

    ann.AddLayer(NumberOfNeurons, ActivationFunction, ActivationFunctionDerivative);
link all layers of neuron together (very important)

    ann.LinkLayers();
    
to train the neural network use the function `ann.Learn(Input, ExpectedOutput, LearningRate)` Example:

    for(int i = 0; i < epoch; i++)
    {
	    float error = ann.Learn(your_input, expected_output, learning_rate);
	    std::cout << "Error: " << error << std::endl;
	}

compute the output of the neural network with `ann.ComputeOutput(input)` Example:

    std::vector<double> output = ann.ComputeOutput(input);
    
    for (unsigned int i = 0; i < output.size(); i++)
    {
    	std::cout << "output[" << i << "]: " << output[i] << std::endl;
    }
