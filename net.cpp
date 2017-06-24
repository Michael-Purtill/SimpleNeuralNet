#include "net.h"
#include <cmath>
#include <iostream>

NeuralNet::NeuralNet(int i, int j, int t)
{
	input[0] = i;
	input[1] = j;
	target = t;
	inputw1 = 0.8;
	inputw2 = 0.4;
	inputw3 = 0.3;
	inputw4 = 0.2;
	inputw5 = 0.9;
	inputw6 = 0.5;
	outw1[0] = 0.3;
	outw2[0] = 0.5;
	outw3[0] = 0.9;
	output[0] = output[1] = -1;
}

float NeuralNet::sigmoid(float n)
{
	return 1 / (1 + (1 / (pow(elog, n))));
}

float NeuralNet::sigmoidPrime(float n)
{
	return (pow(elog, n)) / (pow((1 + pow(elog, n)), 2));
}

void NeuralNet::frontProp()
{
	hidden[0] = input[0] * inputw1 + input[1] * inputw4;
	hidden[1] = input[0] * inputw2 + input[1] * inputw5;
	hidden[2] = input[0] * inputw3 + input[1] * inputw6;

	hidden[3] = sigmoid(hidden[0]);
	hidden[4] = sigmoid(hidden[1]);
	hidden[5] = sigmoid(hidden[2]);

	output[0] = hidden[3] * outw1[0] + hidden[4] * outw2[0] + hidden[5] * outw3[0];
	output[1] = sigmoid(output[0]);

	margin = target - output[1];
}

void NeuralNet::backProp()
{
	deltaOutput = sigmoidPrime(output[0]) * margin;
	deltaWeights[0] = deltaOutput * hidden[3];
	deltaWeights[1] = deltaOutput * hidden[4];
	deltaWeights[2] = deltaOutput * hidden[5];
	outw1[1] = outw1[0] + deltaWeights[0];
	outw2[1] = outw2[0] + deltaWeights[1];
	outw3[1] = outw3[0] + deltaWeights[2];

	deltaHidden[0] = deltaOutput * outw1[0] * sigmoidPrime(hidden[0]);
	deltaHidden[1] = deltaOutput * outw2[0] * sigmoidPrime(hidden[1]);
	deltaHidden[2] = deltaOutput * outw3[0] * sigmoidPrime(hidden[2]);

	deltaWeights[0] = deltaHidden[0] * input[0];
	deltaWeights[1] = deltaHidden[1] * input[0];
	deltaWeights[2] = deltaHidden[2] * input[0];
	deltaWeights[3] = deltaHidden[0] * input[1];
	deltaWeights[4] = deltaHidden[1] * input[1];
	deltaWeights[5] = deltaHidden[2] * input[1];

	inputw1 += deltaWeights[0];
	inputw2 += deltaWeights[1];
	inputw3 += deltaWeights[2];
	inputw4 += deltaWeights[3];
	inputw5 += deltaWeights[4];
	inputw6 += deltaWeights[5];

	outw1[0] = outw1[1];
	outw2[0] = outw2[1];
	outw3[0] = outw3[1];
}

float NeuralNet::outputAnswer()
{
	return output[1];
}

int NeuralNet::outputTarget()
{
	return target;
}
