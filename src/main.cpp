#include "nanoLK.hpp"

int
main()
{
	nanoLK<float> nn(10, 10, 300.0e-9, 300.0e-9);
	std::cout << "Assemble time!\n";
	nn.assemble(0.0f);
	std::cout << "Diagonalization time!\n";
	nn.diagonalize();
}
