#pragma once

#include <mpi.h>

#ifdef COMPLEX
#include <complex>
#endif

// Datatypes from
// MPI: A Message-Passing Interface Standard Version 3.1
// (https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report.pdf)

template <typename T>
struct mpi_type_provider;

// Table 3.2: Predefined MPI datatypes corresponding to C datatypes (page 26)

template <>
struct mpi_type_provider<char> {
  static MPI_Datatype mpi_type() { return MPI_CHAR; }
};

// signed short int is equivalent to int16_t
// template <>
// struct mpi_type_provider<signed short int>
// {
//     static MPI_Datatype mpi_type() { return MPI_SHORT; }
// };

// signed int is equivalent to int32_t
// template <>
// struct mpi_type_provider<signed int>
// {
//     static MPI_Datatype mpi_type() { return MPI_INT; }
// };

// signed long int is equivalent to int64_t
// template <>
// struct mpi_type_provider<signed long int>
// {
//     static MPI_Datatype mpi_type() { return MPI_LONG; }
// };

template <>
struct mpi_type_provider<signed long long int> {
  static MPI_Datatype mpi_type() { return MPI_LONG_LONG; }
};

// signed char is equivalent to int8_t
// template <>
// struct mpi_type_provider<signed char>
// {
//     static MPI_Datatype mpi_type() { return MPI_SIGNED_CHAR; }
// };

// unsigned char is equivalent to uint8_t
// template <>
// struct mpi_type_provider<unsigned char>
// {
//     static MPI_Datatype mpi_type() { return MPI_UNSIGNED_CHAR; }
// };

// unsigned short int is equivalent to uint16_t
// template <>
// struct mpi_type_provider<unsigned short int>
// {
//     static MPI_Datatype mpi_type() { return MPI_UNSIGNED_SHORT; }
// };

// unsigned int is equivalent to uint32_t
// template <>
// struct mpi_type_provider<unsigned int>
// {
//     static MPI_Datatype mpi_type() { return MPI_UNSIGNED; }
// };

// unsigned long int is equivalent to uint64_t
// template <>
// struct mpi_type_provider<unsigned long int>
// {
//     static MPI_Datatype mpi_type() { return MPI_UNSIGNED_LONG; }
// };

template <>
struct mpi_type_provider<unsigned long long int> {
  static MPI_Datatype mpi_type() { return MPI_UNSIGNED_LONG_LONG; }
};

template <>
struct mpi_type_provider<float> {
  static MPI_Datatype mpi_type() { return MPI_FLOAT; }
};

template <>
struct mpi_type_provider<double> {
  static MPI_Datatype mpi_type() { return MPI_DOUBLE; }
};

template <>
struct mpi_type_provider<long double> {
  static MPI_Datatype mpi_type() { return MPI_LONG_DOUBLE; }
};

template <>
struct mpi_type_provider<wchar_t> {
  static MPI_Datatype mpi_type() { return MPI_WCHAR; }
};

// template <>
// struct mpi_type_provider<_Bool>
// {
//     static MPI_Datatype mpi_type() { return MPI_C_BOOL; }
// };

template <>
struct mpi_type_provider<int8_t> {
  static MPI_Datatype mpi_type() { return MPI_INT8_T; }
};

template <>
struct mpi_type_provider<int16_t> {
  static MPI_Datatype mpi_type() { return MPI_INT16_T; }
};

template <>
struct mpi_type_provider<int32_t> {
  static MPI_Datatype mpi_type() { return MPI_INT32_T; }
};

template <>
struct mpi_type_provider<int64_t> {
  static MPI_Datatype mpi_type() { return MPI_INT64_T; }
};

template <>
struct mpi_type_provider<uint8_t> {
  static MPI_Datatype mpi_type() { return MPI_UINT8_T; }
};

template <>
struct mpi_type_provider<uint16_t> {
  static MPI_Datatype mpi_type() { return MPI_UINT16_T; }
};

template <>
struct mpi_type_provider<uint32_t> {
  static MPI_Datatype mpi_type() { return MPI_UINT32_T; }
};

template <>
struct mpi_type_provider<uint64_t> {
  static MPI_Datatype mpi_type() { return MPI_UINT64_T; }
};

template <>
struct mpi_type_provider<float _Complex> {
  static MPI_Datatype mpi_type() { return MPI_C_FLOAT_COMPLEX; }
};

template <>
struct mpi_type_provider<double _Complex> {
  static MPI_Datatype mpi_type() { return MPI_C_DOUBLE_COMPLEX; }
};

template <>
struct mpi_type_provider<long double _Complex> {
  static MPI_Datatype mpi_type() { return MPI_C_LONG_DOUBLE_COMPLEX; }
};

// Table 3.4: Predefined MPI datatypes corresponding to C++ datatypes (page 27)

template <>
struct mpi_type_provider<bool> {
  static MPI_Datatype mpi_type() { return MPI_CXX_BOOL; }
};

#ifdef COMPLEX
template <>
struct mpi_type_provider<std::complex<float>> {
  static MPI_Datatype mpi_type() { return MPI_CXX_FLOAT_COMPLEX; }
};

template <>
struct mpi_type_provider<std::complex<double>> {
  static MPI_Datatype mpi_type() { return MPI_CXX_DOUBLE_COMPLEX; }
};

template <>
struct mpi_type_provider<std::complex<long double>> {
  static MPI_Datatype mpi_type() { return MPI_CXX_LONG_DOUBLE_COMPLEX; }
};
#endif

template <typename T>
MPI_Datatype get_mpi_type() {
  return mpi_type_provider<T>::mpi_type();
}
