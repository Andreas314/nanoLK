#include "nanoLK.hpp"

int
main()
{
	nanoLK<double> nn(8, 8, 30e-9, 30e-9);
	std::cout << "Assemble time!\n";
	nn.assemble(0);
	std::cout << "Diagonalization time!\n";
	nn.diagonalize();
	//nn.write_function(0.05e-9, 0.05e-9, 0);
}
