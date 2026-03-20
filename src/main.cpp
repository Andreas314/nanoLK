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
	
	nanoLK<double> nn(4, 4, 20e-9, 20e-9);
	
	real k_max = 0.2;
	real k_min = -0.2;
	real k_step = 0.05;
	real k_size = k_max - k_min;
	real my_beg = k_size * static_cast<real>(mpi_rank) / static_cast<real>(mpi_size) + k_min;
	real my_end = k_size * static_cast<real>(mpi_rank + 1) / static_cast<real>(mpi_size) + k_min;
	matrixP<double> pp(nn, my_beg, my_end, k_step, mpi_comm, mpi_rank, mpi_size);
	pp.run();
	MPI_Finalize();
}
