#define H_PLANC 1.054571817e-34
#define E_MASS 9.1093837e-31
#define EV_TO_J 1.60217663e-19

#include <vector>
#include <cmath>
#include <complex>
#include <array>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
extern "C" 
{
           void zheev_(char* jobz, char* uplo, int* n,
           std::complex<double>* a, int* lda,
           double* w,
           std::complex<double>* work, int* lwork,
	   double* rwork, int* info);
}
extern "C" 
{
           void cheev_(char* jobz, char* uplo, int* n,
           std::complex<float>* a, int* lda,
           float* w,
           std::complex<float>* work, int* lwork,
	   float* rwork, int* info);
}


template <class T>
class nanoLK
{
public:
	using real = T;
	using ind = int;
	using vec = std::array<ind, 2>;
	nanoLK(ind n_x_, ind n_y_, real l_x_, real l_y_):
		l_x(l_x_),
		l_y(l_y_),
		n_x(n_x_),
		n_y(n_y_)
		{
			size = n_bands * (2 * n_x  + 1) * (2 * n_y + 1);
			hamiltonian.resize( ( size * size ));
			eigenvalues.resize(size);
			G_x = 2.0 * M_PI / l_x;
			G_y = 2.0 * M_PI / l_y;
		}
	void assemble(real );
	void diagonalize();

private:
	constexpr static std::complex<real> i_u = std::complex<real>(0.0, 1.0);
	struct params
	{
		real gamma_l_1 = 7.1 ;
		real gamma_l_2 = 2.01;
		real gamma_l_3 = 2.91;
		real m_c = 0.067;
		real e_p = 28.8*EV_TO_J;
		real delta_so = 0.34 *EV_TO_J;
		real e_g = 1.5 *EV_TO_J;
		real gamma_c =  1.0 / m_c - (e_p / 3.0) *
		 (2.0 / e_g +  1.0 / (e_g + delta_so));
		real gamma_1 = gamma_l_1 - 
			  e_p / (3 * e_g + delta_so);
		real gamma_2 = gamma_l_2 - 
			  e_p / (6 * e_g + 2 * delta_so);
		real gamma_3 = gamma_l_3 - 
			  e_p / (6 * e_g + 2 * delta_so);
		real a = 5e-10 ;
		std::complex<real> p_0 = i_u * static_cast<std::complex<real>>(std::sqrt(e_p) * H_PLANC / 2 / E_MASS);
		real s_x = 100e-9 ;
		real s_y = 100e-9 ;
		real f_mx = 0;
	};

	const params m_params;
	
	constexpr static ind n_bands = 8;
	constexpr static real pre_fact = (H_PLANC * H_PLANC / 2.0 / E_MASS);
	ind n_x, n_y, size;
	real l_x, l_y, G_x, G_y;
	std::vector<std::complex<real>> hamiltonian;
	std::vector<real> eigenvalues;
	
	vec get_global_index(ind , ind , vec , vec ) const;
	std::complex<real> element_right(ind , ind , real , vec , vec ) const;

	real xi_mx(ind , ind) const;
	
	std::complex<real> h0(vec , vec , std::complex<real> ) const;
	std::complex<real> h1(ind , ind , vec , vec , std::complex<real> ) const;
	std::complex<real> h2(ind , ind , ind , ind , vec , vec, std::complex<real> ) const;
	
	std::complex<real> element_o(real , vec , vec ) const;
	std::complex<real> element_p(real , vec , vec ) const;
	std::complex<real> element_q(real , vec , vec ) const;
	std::complex<real> element_r(vec , vec ) const;
	std::complex<real> element_s(real , vec , vec ) const;
	std::complex<real> element_t(vec , vec ) const;
	std::complex<real> element_u(real ) const;
};

template <>
inline void nanoLK<float>::diagonalize()
{
	char flag_eigen = 'V';
	char flag_triangle = 'L';
	int info;
	int lwork = 2 * size;
	
	std::vector<std::complex<float>> work;
	work.resize(lwork);
	
	std::vector<float> rwork;
	rwork.resize(3 * size - 1);

	cheev_(&flag_eigen, &flag_triangle, &size,
        hamiltonian.data(), &size,
        eigenvalues.data(),
        work.data(), &lwork,
	rwork.data(), &info);
	
	if (info != 0)
		throw std::runtime_error("Diagonalization return with info="+std::to_string(info));
	for (int ii = 0; ii < size; ii++)
		std::cout << eigenvalues[ii] / EV_TO_J<< std::endl;
}


template <>
inline void nanoLK<double>::diagonalize()
{
	char flag_eigen = 'V';
	char flag_triangle = 'U';
	int info;
	int lwork = 2 * size;
	
	std::vector<std::complex<double>> work;
	work.resize(lwork);
	
	std::vector<double> rwork;
	rwork.resize(3 * size - 1);

	zheev_(&flag_eigen, &flag_triangle, &size,
        hamiltonian.data(), &size,
        eigenvalues.data(),
        work.data(), &lwork,
	rwork.data(), &info);
	
	if (info != 0)
		throw std::runtime_error("Diagonalization return with info="+std::to_string(info));
}


template <class T>
inline void nanoLK<T>::assemble(real k_z)
{
	for (ind k_x = -n_x; k_x <= n_x; k_x++)
	{
		for (ind k_y = -n_y; k_y <= n_y; k_y++)
		{
			vec k{k_x, k_y};
			for (ind q_x = -n_x; q_x <= n_x; q_x++)
			{
				for (ind q_y = -n_y; q_y <= n_y; q_y++)
				{
					vec q{q_x, q_y};
					for (ind n_b_1 = 0; n_b_1 < n_bands; n_b_1++)
					{
						for (ind n_b_2 = 0; n_b_2 < n_bands; n_b_2++)
						{
							vec index_2d = get_global_index(n_b_1, n_b_2, k, q);
							if (index_2d[0] < index_2d[1])
								continue;
							ind index_1d = index_2d[0] * size + index_2d[1];
							hamiltonian[index_1d] = element_right(n_b_1, n_b_2, k_z, k, q);
						}

					}
				}
			}
		}
	}
}

template <class T>
inline typename nanoLK<T>::vec nanoLK<T>::get_global_index(ind n_b_1, ind n_b_2, vec k, vec q) const
{
	vec index;

	index[0] = n_b_1 * (2 * n_x + 1) * (2 * n_y + 1);
	index[0] += (k[1] + n_y) * (2 * n_x + 1);
	index[0] += (k[0] + n_x);

	index[1] = n_b_2 * (2 * n_x + 1) * (2 * n_y + 1);
	index[1] += (q[1] + n_y) * (2 * n_x + 1);
	index[1] += (q[0] + n_x);

	return index;
}


template <class T>
inline T nanoLK<T>::xi_mx(ind k_x, ind k_y) const
{
	if (k_x != 0 &&  k_y != 0)
	return 1.0 - 4.0 * std::sin(G_x * k_x * m_params.s_x) 
			 * std::sin(G_y * k_y * m_params.s_y)
			 / (G_y * k_y * G_x * k_x);
	else if (k_y == 0 && k_x != 0)
	return 1.0 - 4.0 * std::sin(G_x * k_x * m_params.s_x) 
			 / (G_x * k_x) * m_params.s_y;
	else if (k_x == 0 && k_y != 0)
	return 1.0 - 4.0 * std::sin(G_y * k_y * m_params.s_y) 
			 / (G_y * k_y) * m_params.s_x;
	else
		return 1.0 - 4.0 * m_params.s_x * m_params.s_y;
}

template <class T>
inline std::complex<T> nanoLK<T>::h0(vec k, vec q, std::complex<real> f) const
{
	std::complex<real> result;
	if (k[0] == q[0] && k[1] == q[1])
		result = f;
	result +=  static_cast<std::complex<real>>(std::pow(2 * M_PI, 2.0f)) / l_x / l_y * xi_mx(q[0] - k[0], q[1] - k[1]) * (m_params.f_mx - f);
	return result;
}

template <class T>
inline std::complex<T> nanoLK<T>::h1(ind k_i, ind q_i, vec k, vec q, std::complex<real> f) const
{
	real pre = (k[0] == k_i) ? G_x : G_y;
	return static_cast<std::complex<real>>( 1.0 / 2.0 * pre * (k_i + q_i) ) * h0(k, q, f);

}

template <class T>
inline std::complex<T> nanoLK<T>::h2(ind k_j, ind q_j, ind k_i, ind q_i, vec k, vec q, std::complex<real> f) const
{
	real pre_ki = (k[0] == k_i) ? G_x : G_y;
	real pre_kj = (k[0] == k_j) ? G_x : G_y;
	real pre_qi = (q[0] == q_i) ? G_x : G_y;
	real pre_qj = (q[0] == q_j) ? G_x : G_y;
	return static_cast<std::complex<real>>( 1.0 / 2.0 * (pre_ki * k_i * pre_qj * q_j
			       	               + pre_qi * q_i * pre_kj * k_j) ) * h0(k, q, f);
}

template <class T>
inline std::complex<T> nanoLK<T>::element_o(real k_z, vec k, vec q) const
{
	std::complex<real> f = pre_fact * m_params.gamma_c;
	return f * k_z * k_z + h2(k[0], q[0], k[0], q[0], k, q, f) + h2(k[1], q[1], k[1], q[1], k, q, f);
}

template <class T>
inline std::complex<T> nanoLK<T>::element_p(real k_z, vec k, vec q) const
{
	std::complex<real> f = pre_fact * m_params.gamma_1;
	return f * k_z * k_z + h2(k[0], q[0], k[0], q[0], k, q, f) + h2(k[1], q[1], k[1], q[1], k, q, f);
}

template <class T>
inline std::complex<T> nanoLK<T>::element_q(real k_z, vec k, vec q) const
{
	std::complex<real> f = pre_fact * m_params.gamma_2;
	return f * static_cast<std::complex<real>>(-2.0) *  k_z * k_z + h2(k[0], q[0], k[0], q[0], k, q, f) + h2(k[1], q[1], k[1], q[1], k, q, f);
}

template <class T>
inline std::complex<T> nanoLK<T>::element_r(vec k, vec q) const
{
	std::complex<real> f1 = pre_fact * std::sqrt(3) * m_params.gamma_2;
	std::complex<real> f2 = static_cast<std::complex<real>>(pre_fact * std::sqrt(3) * m_params.gamma_3 * -2.0) * i_u;
	return h2(k[0], q[0], k[0], q[0], k, q, f1) + h2(k[1], q[1], k[1], q[1], k, q, -f1) + h2(k[0], q[0], k[1], q[1], k, q, f2);
}

template <class T>
inline std::complex<T> nanoLK<T>::element_s(real k_z, vec k, vec q) const
{
	std::complex<real> f = pre_fact * std::sqrt(6) * m_params.gamma_3;
	return h1(k[0], q[0], k, q, f) * k_z + h1(k[1], q[1], k, q, f * (-i_u)) * k_z;
}

template <class T>
inline std::complex<T> nanoLK<T>::element_t(vec k, vec q) const
{
	std::complex<real> f = static_cast<std::complex<real>>(1.0 / std::sqrt(6)) * m_params.p_0;
	return h1(k[0], q[0], k, q, f) + h1(k[1], q[1], k, q, f * i_u);
}

template <class T>
inline std::complex<T> nanoLK<T>::element_u(real k_z) const
{
	return static_cast<std::complex<real>>(1.0 / std::sqrt(3) * k_z) * m_params.p_0;
}

template <class T>
inline std::complex<T> nanoLK<T>::element_right(ind n_1, ind n_2, real k_z, vec k, vec q) const
{
	auto is_this = [n_1, n_2](ind x, ind y) -> bool 
		{
    			return ( (n_1 == x) && (n_2 == y) );	
		};

	//zeros
	if (is_this(0, 4) || is_this(0, 5) || is_this(1, 4) || is_this(1, 5) || is_this(2, 6) || is_this(3, 7) ||
	    is_this(4, 0) || is_this(5, 0) || is_this(4, 1) || is_this(5, 1) || is_this(6, 2) || is_this(7, 3))
		return 0.0;
	
	//diagonals
	if (is_this(0,0) || is_this(4,4))
		return m_params.e_g + element_o(k_z, k, q);
	if (is_this(1,1) || is_this(5,5))
		return -(element_p(k_z, k, q) + element_q(k_z, k, q));
	if (is_this(2,2) || is_this(6,6))
		return -(element_p(k_z, k, q) - element_q(k_z, k, q));
	if (is_this(3,3) || is_this(7,7))
		return -(element_p(k_z, k, q) + m_params.delta_so);
	
	//elements with Q
	if (is_this(2, 3) || is_this(3,2) || is_this(6,7) || is_this(7,6))
		return static_cast<real>(-std::sqrt(2)) * element_q(k_z, k, q);
	
	//elements with T
	//with sqrt(3)
	if (is_this(0, 1) || is_this(5,4))
		return static_cast<real>(-std::sqrt(3)) * element_t(k, q);
	if (is_this(1, 0) || is_this(4,5))
		return static_cast<real>(-std::sqrt(3)) * std::conj(element_t(k, q));
	//with(sqrt(2))
	if (is_this(0, 7))
		return static_cast<real>(-std::sqrt(2)) * std::conj(element_t(k, q));
	if (is_this(3, 4))
		return static_cast<real>(std::sqrt(2)) * std::conj(element_t(k, q));
	if (is_this(7, 0))
		return static_cast<real>(-std::sqrt(2)) * element_t(k, q);
	if (is_this(4, 3))
		return static_cast<real>(std::sqrt(2)) * element_t(k, q);
	//without prefactor
	if (is_this(0, 6))
		return -std::conj(element_t(k, q));
	if (is_this(2, 4))
		return std::conj(element_t(k, q));
	if (is_this(6, 0))
		return -element_t(k, q);
	if (is_this(4, 2))
		return element_t(k, q);

	//eleemnts with U
	//with sqrt(2) prefactor
	if (is_this(0, 2) || is_this(6, 4))
		return static_cast<real>(std::sqrt(2)) * element_u(k_z);
	if (is_this(2, 0) || is_this(4, 6))
		return static_cast<real>(std::sqrt(2)) * std::conj(element_u(k_z));
	//with no prefactor
	if (is_this(0, 3) || is_this(7, 4))
		return -element_u(k_z);
	if (is_this(3, 0) || is_this(4, 7))
		return -std::conj(element_u(k_z));

	//elements with S
	//with sqrt(3)
	if (is_this(2, 7))
		return static_cast<real>(std::sqrt(3)) * element_s(k_z, k, q);
	if (is_this(3, 6))
		return static_cast<real>(-std::sqrt(3)) * element_s(k_z, k, q);
	if (is_this(7, 2))
		return static_cast<real>(std::sqrt(3)) * std::conj(element_s(k_z, k, q));
	if (is_this(6, 3))
		return static_cast<real>(-std::sqrt(3)) * std::conj(element_s(k_z, k, q));
	//with sqrt(2)
	if (is_this(1, 2) || is_this(6, 5))
		return static_cast<real>(std::sqrt(2)) * element_s(k_z, k, q);
	if (is_this(2, 1) || is_this(5, 6))
		return static_cast<real>(std::sqrt(2)) * std::conj(element_s(k_z, k, q));
	//with no prefactor
	if (is_this(1, 3) || is_this(7, 5))
		return  -element_s(k_z, k, q);
	if (is_this(3, 1) || is_this(5, 7))
		return  -std::conj(element_s(k_z, k, q));
	
	//elements with R
	//with sqrt(2)
	if (is_this(7, 1))
		return -static_cast<real>(std::sqrt(2)) * std::conj(element_r(k, q));
	if (is_this(5, 3))
		return static_cast<real>(std::sqrt(2)) * std::conj(element_r(k, q));
	if (is_this(1, 7))
		return -static_cast<real>(std::sqrt(2)) * element_r(k, q);
	if (is_this(3, 5))
		return static_cast<real>(std::sqrt(2)) * element_r(k, q);
	//with no prefactor
	if (is_this(6, 1))
		return -std::conj(element_r(k, q));
	if (is_this(5, 2))
		return std::conj(element_r(k, q));
	if (is_this(1, 6))
		return -element_r(k, q);
	if (is_this(2, 5))
		return element_r(k, q);
	return 0;


}
