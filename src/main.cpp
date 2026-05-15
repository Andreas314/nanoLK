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
	int num_steps_per_proc = std::atoi(argv[3]);
	int num_steps = num_steps_per_proc * mpi_size;
	real k_size = (k_max - k_min) / mpi_size;
	real my_beg = k_min + mpi_rank * k_size;
	real my_end = k_min + (mpi_rank + 1) * k_size;
	double L = std::atoi(argv[2]) * 1e-9;
	int N = std::atoi(argv[1]);
	if (mpi_rank == 0 )
		std::cout << "Setting up simulation with L = " << L * 1e9 << " nm and N = " << N << "\n";
	nanoLK<double> nn(N, N, L, L);
	matrixP<double> pp(nn, my_beg, my_end, num_steps_per_proc, 0.79, 0.88, 50, mpi_comm, mpi_rank, mpi_size);
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
