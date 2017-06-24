class NeuralNet
{
private:
	float inputw1;
	float inputw2;
	float inputw3;
	float inputw4;
	float inputw5;
	float inputw6;
	float outw1[2];
	float outw2[2];
	float outw3[2];
	int input[2];
	float hidden[6];
	float output[2];
	int target;
	float margin;
	float deltaOutput;
	float deltaWeights[6];
	float deltaHidden[3];
	double elog = 2.7182818284590452353602874713526624977572470937;
public:
	NeuralNet(int i, int j, int t);
	float sigmoid(float n);
	float sigmoidPrime(float n);
	void frontProp();
	void backProp();
	float outputAnswer();
};
