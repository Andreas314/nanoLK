#include "nanoLK.hpp"
#include "matrixP.hpp"
#include <mpi.h>
int
main(int argc, char** argv)
{
	//MPI initialization
      	MPI_Init(&argc, &argv);
        MPI_Comm mpi_comm = MPI_COMM_WORLD;
        int mpi_size;
        MPI_Comm_size(mpi_comm, &mpi_size);
        int mpi_rank;
        MPI_Comm_rank(mpi_comm, &mpi_rank);

	//actual code
	using real = double; 
	
	real k_max = 0.2;
	real k_min = -0.2;
	real k_step = 0.02;
	real k_size = k_max - k_min;
	real my_beg = k_size * static_cast<real>(mpi_rank) / static_cast<real>(mpi_size) + k_min;
	real my_end = k_size * static_cast<real>(mpi_rank + 1) / static_cast<real>(mpi_size) + k_min;
	if (mpi_rank + 1 == mpi_size)
		my_end = k_max;
	for (real k = my_beg; k < my_end; k+=k_step)
	{
		if (k < 1e-5)
			k += 1e-5;
		nanoLK<double> nn(6, 6, 18e-9, 18e-9);
		std::cout << "Assembling!" << std::endl;
		//nn.assemble(k);
		nn.assemble(1e-5);
		std::cout << "Diagonalizing!" << std::endl;
		nn.diagonalize();
		std::cout << "Writing!" << std::endl;
		nn.write_functions(0.05e-9,0.05e-9,-1, true);
	}

//	matrixP<double> pp(nn, my_beg, my_end, k_step, mpi_comm, mpi_rank, mpi_size);
//	pp.run();
	MPI_Finalize();
}
