#ifndef MAJORANA_DISORDER_PARAMETER_H__ 
#define MAJORANA_DISORDER_PARAMETER_H__

#include <string>
#include <boost/multi_array.hpp>
#include <boost/random/mersenne_twister.hpp>
#include "defs.h"

class MajoranaDisorderParameter {
public:
    virtual ~MajoranaDisorderParameter();

    // Access J(i,j,k,l) (0-based)
    virtual double elem(int i, int j, int k, int l) = 0;
};

/*
 * Holds the J_{ijkl} parameters in the convention where the 
 * Hamiltonian ordering is:
 *
 * H ~ J_{ijkl} \chi_i \chi_j \chi_k \chi_l
 *
 * where $\chi_i are Majorana fermions. J_{ijkl} real and is completely
 * antisymmetric. The elements are Gaussian distributed with
 * \sigma^2 = 3! J^2 / N^3 .
 */
class MajoranaKitaevDisorderParameter : public MajoranaDisorderParameter {
public:
    // Create the object with given N and J parameters, and given
    // random number generator 'gen'.
    MajoranaKitaevDisorderParameter(
            int N, double J, boost::random::mt19937* gen);

    // Initializes to zero
    MajoranaKitaevDisorderParameter(int N);

    virtual ~MajoranaKitaevDisorderParameter();

    // Access J(i,j,k,l) (0-based)
    virtual double elem(int i, int j, int k, int l);

    void antisymmetrize();

    string to_string();

    const int N;
    boost::multi_array<double, 4> Jelems;
};

// J(i,j,k,l) = i-j+k-l
class MockMajoranaDisorderParameter : public MajoranaDisorderParameter {
public:
    virtual ~MockMajoranaDisorderParameter();
    virtual double elem(int i, int j, int k, int l);
};

// A disorder parameter that does not include neighboring Majoranas that translate
// to the same Dirac fermion. Useful for testing GPU code.
class MajoranaKitaevDisorderParameterWithoutNeighbors
    : public MajoranaKitaevDisorderParameter {
public:
    // Create the object with given N and J parameters, and given
    // random number generator 'gen'.
    MajoranaKitaevDisorderParameterWithoutNeighbors(
            int N, double J, boost::random::mt19937* gen);

    virtual ~MajoranaKitaevDisorderParameterWithoutNeighbors();
};

#endif // MAJORANA_DISORDER_PARAMETER_H__

