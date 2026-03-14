#include "nanoLK.hpp"

int
main()
{
	nanoLK<float> nn(8, 8, 55.0e-9, 55.0e-9);
	std::cout << "Assemble time!\n";
	nn.assemble(0);
	std::cout << "Diagonalization time!\n";
	nn.diagonalize();
}
