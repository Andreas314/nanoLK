#pragma once
#define f_2 1/sqrt(2)
#define f_3 1/sqrt(3)
#define f_6 1/sqrt(6)

#include "nanoLK.hpp"

#include <set>
#include <mpi.h>

template<class T>
class matrixP
{
	public:
		using tensor4D = std::array<std::array<std::array<std::array<std::complex<T>, 3>, 3>, 3>, 3>;
		matrixP(nanoLK<T> &hamiltonian_, T k_z_min_, T k_z_max_, T k_z_step_, MPI_Comm& mpi_comm_, int mpi_rank_, int mpi_size_):
			hamiltonian(hamiltonian_),
			k_z_min(k_z_min_),
			k_z_max(k_z_max_),
			k_z_step(k_z_step_),
			mpi_comm(mpi_comm_),
			mpi_rank(mpi_rank_),
			mpi_size(mpi_size_)
			{
				assemble_px();
				assemble_py();
				assemble_pz();
				P = hamiltonian.get_P();
			};
	void run();
	void write_to_file();
	private:
		int mpi_rank, mpi_size;
		MPI_Comm& mpi_comm;
		std::complex<T> i_u{0, 1};
		
		T P;
		T k_z_min, k_z_max, k_z_step;
		nanoLK<T> &hamiltonian;
		std::vector<int> states, valence_states, conduction_states;
		std::complex<T> get_momentum(int, int, int);
		std::complex<T> get_qi_element(int, int, int, int);
		constexpr static int n_bands = 8;
		
		
		void to_complex(std::array<T, 4 * n_bands * n_bands> &inp, std::array<std::array<std::complex<T>, n_bands>, n_bands> &output);
		void assemble_px();
		void assemble_py();
		void assemble_pz();
		void copy_other_half(std::array<T, 4 * n_bands * n_bands> &values);
		std::array<std::array<std::complex<T>, n_bands>, n_bands> p_z, p_y, p_x;
		tensor4D QItensor;
	
};
template <>
void matrixP<double>::run()
{
	using T = double;
	for (T k = k_z_min; k < k_z_max - k_z_step / 2.0; k += k_z_step)
	{
		hamiltonian.assemble(k);
		std::cout << "Diagonalize on " << mpi_rank << " with k_z = " << k  << std::endl;
		hamiltonian.diagonalize();
		states = hamiltonian.get_indices();
		valence_states = hamiltonian.get_valence_states();
		conduction_states = hamiltonian.get_conduction_states();
		std::cout << "Sum on " << mpi_rank << " with k_z = " << k  << std::endl;
		#pragma omp parallel for
		for (int ind = 0; ind < 81; ind++)
		{
			int ind_1 = ind / 27;
			int ind_2 = (ind % 27) / 9;
			int ind_3 = ((ind % 27) % 9) / 3;
			int ind_4 = ((ind % 27) % 9) % 3;
			if (mpi_rank == 0)
				std::cout << ind / 81.0 << "\n";
			QItensor[ind_1][ind_2][ind_3][ind_4] = get_qi_element(ind_1, ind_2, ind_3, ind_4);
		}
	}
	if (mpi_rank == 0) 
	{
   		MPI_Reduce(MPI_IN_PLACE, QItensor.data(), 81, MPI_C_DOUBLE_COMPLEX, MPI_SUM, 0, mpi_comm);
	} 
	else 
	{
   		 MPI_Reduce(QItensor.data(), nullptr , 81, MPI_C_DOUBLE_COMPLEX, MPI_SUM, 0, mpi_comm);
	}
	if (mpi_rank == 0)
		std::cout << QItensor[0][0][0][0];
}

template <>
void matrixP<float>::run()
{
	using T = float;
	for (T k = k_z_min; k < k_z_max - k_z_step / 2.0; k += k_z_step)
	{
		hamiltonian.assemble(k);
		std::cout << "Diagonalize on " << mpi_rank << " with k_z = " << k  << std::endl;
		hamiltonian.diagonalize();
		states = hamiltonian.get_indices();
		valence_states = hamiltonian.get_valence_states();
		conduction_states = hamiltonian.get_conduction_states();
		std::cout << "Sum on " << mpi_rank << " with k_z = " << k  << std::endl;
		#pragma omp parallel for
		for (int ind = 0; ind < 81; ind++)
		{
			int ind_1 = ind / 27;
			int ind_2 = (ind % 27) / 9;
			int ind_3 = ((ind % 27) % 9) / 3;
			int ind_4 = ((ind % 27) % 9) % 3;
			QItensor[ind_1][ind_2][ind_3][ind_4] = get_qi_element(ind_1, ind_2, ind_3, ind_4);
		}
	}
	if (mpi_rank == 0) 
	{
   		MPI_Reduce(MPI_IN_PLACE, QItensor.data(), 81, MPI_C_FLOAT_COMPLEX, MPI_SUM, 0, mpi_comm);
	} 
	else 
	{
   		 MPI_Reduce(QItensor.data(), nullptr , 81, MPI_C_FLOAT_COMPLEX, MPI_SUM, 0, mpi_comm);
	}
}

template <class T>
std::complex<T> matrixP<T>::get_qi_element(int ind_1, int ind_2, int ind_3, int ind_4)
{
	std::complex<T> result = 0;
	T omega_0 = 0.8 / EV_TO_J / H_PLANC;
	T sigma = 0.3 / EV_TO_J / H_PLANC;
	auto gauss = [omega_0, sigma](T omega) -> T
	{
   		return 1.0 / std::sqrt(2 * M_PI) / sigma * exp(-0.5 * std::pow((2 * omega_0 - omega) / sigma, 2));
	};
	for (auto &cond : conduction_states)
	{
		auto p_cc = get_momentum(cond, cond, ind_1);
		for (auto &val : valence_states)
		{
			auto p_vv = get_momentum(val, val, ind_1);
			auto p_vc = get_momentum(val, cond, ind_2);
			T omega_cv = (hamiltonian.get_energy(cond) - hamiltonian.get_energy(val)) / H_PLANC;
			T weight = gauss(omega_cv);
			for (auto &inter : states)
			{
				T omega_ci = (hamiltonian.get_energy(cond) - hamiltonian.get_energy(inter)) / H_PLANC;
				auto p_iv = get_momentum(inter, val, ind_3);
				auto p_ci = get_momentum(cond, inter, ind_4);
				
				auto p_ic = get_momentum(inter, cond, ind_3);
				auto p_vi = get_momentum(val, inter, ind_4);
				result += (p_cc - p_vv) * p_vc * (p_ci * p_iv + p_vi * p_ic) / (omega_0 + omega_ci) * weight;
			}
		}
	}
	//std::cout << result << std::endl;
	return result;


}




template <class T>
void matrixP<T>::to_complex(std::array<T, 4 * n_bands * n_bands> &inp, std::array<std::array<std::complex<T>, n_bands>, n_bands> &output)
{
	for (int ii = 0; ii < n_bands; ++ii)
	{
		for (int jj = 0; jj < n_bands; jj++)
		{
			double real_part = inp[ii * 2 * n_bands + jj];
			double imaginary_part = inp[(ii + n_bands) *2 * n_bands + jj];
			output[ii][jj] = real_part - i_u * imaginary_part;
		}
	}
}

template <class T>
std::complex<T> matrixP<T>::get_momentum(int state_1, int state_2, int direction)
{
	std::complex<T> interband = - i_u * static_cast<T>(H_PLANC) * hamiltonian.integrate_state_and_derivative(state_1, state_2, direction);
	std::complex<T> intraband= 0.0;
	std::array<std::array<std::array<std::complex<T>, n_bands>, n_bands>*, 3> ps{&p_x, &p_y, &p_z};
	auto p = ps[direction];
	for (int ii = 0; ii < n_bands; ii++)
	{
		for (int jj = ii; jj < n_bands; jj++)
		{
			std::complex<T> res;
			if  ( std::abs((*p)[ii][jj]) != 0)
			{
				res = P * ( (*p)[ii][jj] * 
					hamiltonian.integrate_state_and_state(state_1, state_2, ii, jj));
				intraband += res;
				if (ii != jj)
					intraband += std::conj(res);
			}
		}
	}
	return intraband + interband;


}

template <class T>
void matrixP<T>::assemble_px()
{
	std::array<T, 4 * n_bands * n_bands> px;
	std::set<int> indices = {9, 14, 15, 24, 44, 60, 74, 75, 77, 92, 104, 120};
	std::vector<double> values = {-f_2, - f_6, - f_3, f_2, - f_6, - f_3, f_6, f_3, f_2, - f_2, f_6, f_3};
	int counter = 0;
	for (int ii = 0; ii < 2 * n_bands * n_bands; ++ii)
	{
		if (indices.contains(ii))
		{
			px[ii] = values[counter];
			++counter;
		}
		else
		{
			px[ii] = 0;
		}
	}
	copy_other_half(px);
	to_complex(px, p_x);
}

template <class T>
void matrixP<T>::assemble_py(){
	std::array<T, 4 * n_bands * n_bands> py;
	std::set<int> indices = {1, 6, 7, 16, 36, 52, 66, 67, 69, 84, 96, 112};
	std::vector<double> values = {-f_2, f_6, f_3, - f_2, f_6, f_3, f_6, f_3, f_2, f_2, f_6, f_3};
	int counter = 0;
	for (int ii = 0; ii < 2 * n_bands * n_bands; ++ii)
	{
		if (indices.contains(ii))
		{
			py[ii] = values[counter];
			++counter;
		}
		else
		{
			py[ii] = 0;
		}
	}
	copy_other_half(py);
	to_complex(py, p_y);
}
template <class T>
void matrixP<T>::assemble_pz(){
	std::array<T, 4 * n_bands * n_bands> pz;
	std::set<int> indices = {10, 11, 40, 56, 78, 79, 108, 124};
	std::vector<double> values = {2 * f_6, - f_3, - 2 * f_6, f_3, 2 * f_6, - f_3, - 2 * f_6, f_3};
	int counter = 0;
	for (int ii = 0; ii < 2 * n_bands * n_bands; ++ii){
		if (indices.contains(ii)){
			pz[ii] = values[counter];
			++counter;
		}
		else{
			pz[ii] = 0;
		}
	}
	copy_other_half(pz);
	to_complex(pz, p_z);
}

template <class T>
void matrixP<T>::copy_other_half(std::array<T, 4 * n_bands * n_bands> &values){
	for (int ii = 2 * n_bands * n_bands; ii < 4 * n_bands * n_bands; ++ii){
		if ((ii / n_bands) % 2 == 0)
		{
			values[ii] = -values[ii - 2 * n_bands * n_bands + n_bands];
		}
		else{
			values[ii] = values[ii - 2 * n_bands * n_bands - n_bands];
		}
	}
}
