#pragma once
#define f_2 1/sqrt(2)
#define f_3 1/sqrt(3)
#define f_6 1/sqrt(6)
#define EV_TO_J 1.60217663e-19
#define H_PLANC 1.054571817e-34
#define E_MASS 9.1093837e-31

#include "nanoLK.hpp"

#include <set>
#include <mpi.h>

template<class T>
class matrixP
{
	public:
		using tensor4D = std::array<std::array<std::array<std::array<std::complex<T>, 3>, 3>, 3>, 3>;
		matrixP(nanoLK<T> &hamiltonian_, T k_z_min_, T k_z_max_, int k_z_step_,T omega_min_, T omega_max_, int n_steps_,  MPI_Comm& mpi_comm_, int mpi_rank_, int mpi_size_):
			hamiltonian(hamiltonian_),
			k_z_min(k_z_min_),
			k_z_max(k_z_max_),
		    num_step(k_z_step_),
			mpi_comm(mpi_comm_),
			mpi_rank(mpi_rank_),
			mpi_size(mpi_size_),
			omega_min(omega_min_),
			omega_max(omega_max_),
			n_steps(n_steps_)
			{
				assemble_px();
				assemble_py();
				assemble_pz();
				P = hamiltonian.get_P();
				auto opt = hamiltonian.get_grid_info();
				res_x = static_cast<int>(opt[0]);
				res_y = static_cast<int>(opt[1]);
				s_x = opt[2];
				s_y = opt[3];
				l_x = opt[4];
				l_y = opt[5];
				omega_min *= EV_TO_J;
				omega_max *= EV_TO_J;
				T domega = (omega_max - omega_min) / n_steps;
				omegas.resize(n_steps);
				QItensor.resize(n_steps);
				for (int step = 0; step < n_steps; step++)
				{
					omegas[step] = (omega_min + domega * step) / H_PLANC;
					QItensor[step] = 0;
				}

			};
	void run();
	void write_to_file();
	private:
		int mpi_rank, mpi_size;
		MPI_Comm& mpi_comm;
		std::complex<T> i_u{0, 1};
		
		T P;
		T k_z_min, k_z_max;
		int num_step;
		nanoLK<T> &hamiltonian;
		std::vector<std::vector<std::vector<std::vector<std::complex<T>>>>> functions, derivative_x, derivative_y;
		std::vector<int> states, valence_states, conduction_states;
		void pre_evaluate();
		std::complex<T> get_momentum(int, int, int);
		std::complex<T> get_momentum_analytically(int, int, int);
		void get_qi_element(int, int, int, int);
		constexpr static int n_bands = 8;
		int res_x, res_y, n_steps;
		T s_x, s_y, omega_min, omega_max, l_x, l_y;
		std::vector<T> omegas;
		
		T lattice = 5.56e-10;
		void to_complex(std::array<T, 4 * n_bands * n_bands> &inp, std::array<std::array<std::complex<T>, n_bands>, n_bands> &output);
		void assemble_px();
		void assemble_py();
		void assemble_pz();
		void copy_other_half(std::array<T, 4 * n_bands * n_bands> &values);
		std::array<std::array<std::complex<T>, n_bands>, n_bands> p_z, p_y, p_x;
		std::vector<std::complex<T>> QItensor;
	
};
template <>
void matrixP<double>::run()
{
	using T = double;
	T k_z_step =  (k_z_max - k_z_min) / num_step;
	for (int ii = 0; ii < num_step; ++ii)
	{
	    T k = k_z_min + k_z_step * ii;
		hamiltonian.assemble(k);
		//std::cout << "Diagonalize on " << mpi_rank << " with k_z = " << k  << std::endl;
		hamiltonian.diagonalize();
		states = hamiltonian.get_indices();
		valence_states = hamiltonian.get_valence_states();
		conduction_states = hamiltonian.get_conduction_states();
		functions.assign(
		    states.size(),
		    std::vector<std::vector<std::vector<std::complex<T>>>>(
		        8,
		        std::vector<std::vector<std::complex<T>>>(
		            res_y,
		            std::vector<std::complex<T>>(res_x)
		        )
		    )
		);
		derivative_x.assign(
		    states.size(),
		    std::vector<std::vector<std::vector<std::complex<T>>>>(
		        8,
		        std::vector<std::vector<std::complex<T>>>(
		            res_y,
		            std::vector<std::complex<T>>(res_x)
		        )
		    )
		);
		derivative_y.assign(
		    states.size(),
		    std::vector<std::vector<std::vector<std::complex<T>>>>(
		        8,
		        std::vector<std::vector<std::complex<T>>>(
		            res_y,
		            std::vector<std::complex<T>>(res_x)
		        )
		    )
		);
		pre_evaluate();
		//std::cout << "Sum on " << mpi_rank << "v = " << valence_states.size() << " c = " << conduction_states.size() << " t = " << states.size() << "\n";
		get_qi_element(1, 0, 0, 1);
	}
//	std::cout << mpi_rank << " done!\n";
	MPI_Barrier(mpi_comm);

	if (mpi_rank == 0)
	{
	    MPI_Reduce(MPI_IN_PLACE, QItensor.data(), n_steps,
	               MPI_C_DOUBLE_COMPLEX, MPI_SUM, 0, mpi_comm);
	}
	else
	{
	    MPI_Reduce(QItensor.data(), QItensor.data(), n_steps,
	               MPI_C_DOUBLE_COMPLEX, MPI_SUM, 0, mpi_comm);
	}
	if (mpi_rank == 0)
		for(int step = 0; step < n_steps; step++)
		{
			std::cout << omegas[step] / EV_TO_J * H_PLANC << " " << std::abs(QItensor[step]) * k_z_step * M_PI / lattice << "\n";
		}
	}

template <class T>
void matrixP<T>::pre_evaluate()
{
	T dx = s_x / (res_x - 1);
	T dy = s_y / (res_y - 1);
	for (int band = 0; band < 8; band++)
	{
		int counter = 0;
		for (int &state: states)
		{
			for (int n_x = 0; n_x < res_x; n_x++)
			{
				T x = -s_x / 2.0 + n_x * dx;	
				for (int n_y = 0; n_y < res_y; n_y++)
				{
					T y = -s_y / 2.0 + n_y * dy;
					std::complex<T> value = hamiltonian.get_value_at_point(state, band, x, y, 0);
					std::complex<T> dvalue_dx = hamiltonian.get_derivative_at_point(state, band, 0, x, y, 0);
					std::complex<T> dvalue_dy = hamiltonian.get_derivative_at_point(state, band, 1, x, y, 0);
					functions[counter][band][n_y][n_x] = value;
					derivative_x[counter][band][n_y][n_x] = dvalue_dx;
					derivative_y[counter][band][n_y][n_x] = dvalue_dy;
				}

			}
		counter++;
		}
	}
}

template <class T>
void matrixP<T>::get_qi_element(int ind_1, int ind_2, int ind_3, int ind_4)
{
    std::complex<T> delta = i_u * 2.5e-3 * EV_TO_J / H_PLANC;
    std::complex<T> prefactor = std::pow(EV_TO_J / E_MASS, (T)4) 
                              / std::pow(H_PLANC, (T)3) 
                              / s_x / s_y * i_u * 2.0 / (T)valence_states.size() ;

    // Build momentum matrices first
    std::array<std::vector<std::vector<std::complex<T>>>, 3> p;
    for (int ind = 0; ind < 3; ind++)
        p[ind].assign(states.size(), 
                      std::vector<std::complex<T>>(states.size(), 0));
    for (int a = 0; a < states.size(); a++)
    {
        for (int b = a; b < states.size(); b++)
        {
            for (int ind = 0; ind < 3; ind++)
            {
                std::complex<T> res = get_momentum(b, a, ind);
                p[ind][b][a] = res;
                p[ind][a][b] = std::conj(res);
            }
        }
        }

    // Build correct conduction/valence lookup by state index
    // conduction_states and valence_states contain indices INTO states[]
    std::vector<bool> is_cond(states.size(), false);
    std::vector<bool> is_val(states.size(), false);
    for (int idx : conduction_states) is_cond[idx] = true;
    for (int idx : valence_states)    is_val[idx]  = true;

    int N = states.size();

    for (int k = 0; k < N; k++)
    for (int l = 0; l < N; l++)
    for (int q = 0; q < N; q++)
    for (int v = 0; v < N; v++)
    {
        if (!is_val[v]) continue;  // v must be valence

        T E_k = hamiltonian.get_energy(states[k]);
        T E_l = hamiltonian.get_energy(states[l]);
        T E_q = hamiltonian.get_energy(states[q]);
        T E_v = hamiltonian.get_energy(states[v]);

        // All transition frequencies in rad/s
        T w_qv = (E_q - E_v) / H_PLANC;
        T w_lv = (E_l - E_v) / H_PLANC;
        T w_vk = (E_v - E_k) / H_PLANC;
        T w_qk = (E_q - E_k) / H_PLANC;
        T w_lq = (E_l - E_q) / H_PLANC;
        T w_vq = -w_qv;
        T w_vl = -w_lv;
        T w_kv = -w_vk;

        // Numerators exactly from formula (3.5)
        std::array<std::complex<T>, 8> nums;
        nums[0] = p[ind_1][k][l] * p[ind_2][l][q] * p[ind_3][q][v] * p[ind_4][v][k];
        nums[1] = p[ind_1][k][l] * p[ind_2][l][q] * p[ind_3][q][v] * p[ind_4][v][k];
        nums[2] = p[ind_1][k][l] * p[ind_3][l][v] * p[ind_4][v][q] * p[ind_2][q][k];
        nums[3] = p[ind_1][k][l] * p[ind_3][l][v] * p[ind_4][v][q] * p[ind_2][q][k];
        nums[4] = p[ind_1][v][k] * p[ind_2][k][q] * p[ind_3][q][l] * p[ind_4][l][v];
        nums[5] = p[ind_1][k][q] * p[ind_2][q][v] * p[ind_3][v][l] * p[ind_4][l][k];
        nums[6] = p[ind_1][k][q] * p[ind_3][q][l] * p[ind_4][l][v] * p[ind_2][v][k];
        nums[7] = p[ind_1][k][v] * p[ind_3][v][l] * p[ind_4][l][q] * p[ind_2][q][k];
       // for (int ii = 0; ii < 8; ii++)
        //nums[ii] = 1;
        for (int step = 0; step < n_steps; step++)
        {
            T w = omegas[step];

            // Denominators exactly from formula (3.5)
            std::array<std::complex<T>, 8> denoms;
            denoms[0] = -(-w + w_qv + delta) * (w_qk - 2*w + delta);
            denoms[1] = -(-w + w_vk + delta) * (w_qk - 2*w + delta);
            denoms[2] =  (-w + w_lv + delta) * (w_lq - 2*w + delta);
            denoms[3] =  (-w + w_vq + delta) * (w_lq - 2*w + delta);
            denoms[4] =  (-w + w_lv + delta) * (w_qv - 2*w + delta);
            denoms[5] =  (-w + w_vl + delta) * (w_vq - 2*w + delta);
            denoms[6] = -(-w + w_lv + delta) * (w_qv - 2*w + delta);
            denoms[7] = -(-w + w_vl + delta) * (w_vq - 2*w + delta);

            std::complex<T> result = 0;
            for (int cc = 0; cc < 8; cc++)
                result += nums[cc] / denoms[cc];

            QItensor[step] += result * prefactor / std::pow(w, (T)3);
        }
    }
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
//Lambda for <F_n^j|F_m^i>
	auto integrate_func = [this, state_1, state_2](int band_a, int band_b) {
  		T dx = this->s_x / this->res_x;
  		T dy = this->s_y / this->res_y;
		std::complex<T> result = 0;
		for (int nx = 0; nx < res_x; nx++)
		{
			for (int ny = 0; ny < res_y; ny++)
			{
				result += std::conj(this->functions[state_1][band_a][ny][nx]) *
						this->functions[state_2][band_b][ny][nx] /
						std::sqrt(this->hamiltonian.get_norm(states[state_1]) *
						this->hamiltonian.get_norm(states[state_2]));
			}
		}
		result *= (dx * dy);
		return result;
	};

//Lambda for <F_n^j|p_dir|F_m^i>
	auto integrate_der = [this, direction, state_1, state_2](int band_a, int band_b) {
  		T dx = this->s_x / this->res_x;
  		T dy = this->s_y / this->res_y;
		std::complex<T> result = 0;
		for (int nx = 0; nx < res_x; nx++)
		{
			for (int ny = 0; ny < res_y; ny++)
			{
				if (direction == 0)
					result += i_u * std::conj(this->functions[state_1][band_a][ny][nx]) *
						this->derivative_x[state_2][band_b][ny][nx] /
					std::sqrt(this->hamiltonian.get_norm(states[state_1]) *
						this->hamiltonian.get_norm(states[state_2]));
				else
					result += std::conj(this->functions[state_1][band_a][ny][nx]) *
						this->derivative_y[state_2][band_b][ny][nx] /
					std::sqrt(this->hamiltonian.get_norm(states[state_1]) *
						this->hamiltonian.get_norm(states[state_2]));
			}
		}
		result *= (dx * dy);
		return result;
	};
	
	std::complex<T> intraband= 0.0;
	std::complex<T> interband= 0.0;
	std::array<std::array<std::array<std::complex<T>, n_bands>, n_bands>*, 3> ps{&p_x, &p_y, &p_z};
	auto p = ps[direction];
	for (int ii = 0; ii < n_bands; ii++)
	{
		//if (state_1 != state_2)
			intraband += -i_u * static_cast<T>(H_PLANC)*integrate_der(ii, ii);
		for (int jj = ii; jj < n_bands; jj++)
		{
			std::complex<T> res;
			if  ( std::abs((*p)[ii][jj]) != 0)
			{
					interband += P * integrate_func(ii, jj) * (*p)[ii][jj];
				if (!(ii == jj))
					interband += P * std::conj(integrate_func(ii, jj) * ((*p)[ii][jj]));
			}
		}
	}
	return (intraband + interband);


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
