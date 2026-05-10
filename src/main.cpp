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
	real my_beg = k_size * static_cast<real>(mpi_rank) / static_cast<real>(mpi_size) + k_min + 0.01;
	real my_end = k_size * static_cast<real>(mpi_rank + 1) / static_cast<real>(mpi_size) + k_min + 0.01;
	double L = std::atoi(argv[2]) * 1e-9;
	int N = std::atoi(argv[1]);
	if (mpi_rank == 0 )
	{
		std::cout << "Setting up simulation with L = " << L * 1e9 << " nm and N = " << N << "\n";
		my_beg = k_min;
	}
	if (mpi_rank + 1 == mpi_size)
		my_end = k_max;
	
	nanoLK<double> nn(N, N, L, L);
	matrixP<double> pp(nn, my_beg, my_end, k_step, 0.8, 0.9, 10, mpi_comm, mpi_rank, mpi_size);
	pp.run();

//	for (real k = my_beg; k <= my_end; k+=k_step)
//	{
//		if (k < 1e-5)
//			k += 1e-5;
//		nanoLK<double> nn(N, N, L, L);
//		std::cout << "Working on " << k << " from " << my_beg << " to " << my_end << "\n";
//		nn.assemble(k);
//		nn.diagonalize();
//	}

	MPI_Finalize();
}
