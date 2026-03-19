#include "nanoLK.hpp"

int
main()
{
	nanoLK<double> nn(4, 4, 20e-9, 20e-9);
	std::cout << "Assemble time!\n";
	nn.assemble(0);
	std::cout << "Diagonalization time!\n";
	nn.diagonalize();
	auto states = nn.get_indices();
	std::cout << nn.integrate_state_and_derivative(states[0], states[0], 0);
	//nn.write_functions(0.05e-9, 0.05e-9, 2, 8);
}
