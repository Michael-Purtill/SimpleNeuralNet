#include <iostream>
#include "net.h"
#include <cmath>
using namespace std;

int main()
{
	NeuralNet oneOne = oneOne(1, 1, 0);

	while (oneOne.outputAnswer() != 0.0)
	{
		oneOne.frontProp();
		oneOne.backProp();
		cout << oneOne.outputAnswer() << "\n";
	}

	cout << oneOne.outputAnswer() << "\n";

	return 0;
}
