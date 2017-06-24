#include <iostream>
#include "net.h"
#include <cmath>
using namespace std;

int main()
{
	NeuralNet oneOne(1, 1, 0);

	while (oneOne.outputAnswer() != oneOne.outputTarget())
	{
		oneOne.frontProp();
		cout << oneOne.outputAnswer() << '\n';
		oneOne.backProp();
	}

	cout << oneOne.outputAnswer() << '\n';

	return 0;
}