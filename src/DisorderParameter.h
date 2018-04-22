#ifndef DISORDER_PARAMETER_H__ 
#define DISORDER_PARAMETER_H__

#include <boost/multi_array.hpp>
#include <boost/random/mersenne_twister.hpp>
#include "defs.h"

class DisorderParameter {
public:
    virtual ~DisorderParameter();

    // Access J(i,j,k,l) (0-based)
    virtual cpx elem(int i, int j, int k, int l) = 0;
};

/*
 * Holds the J_{ij;kl} parameters in the convention where the Hamiltonian 
 * ordering is:
 *
 * H ~ J_{ij;kl} c^\dagger_i c^\dagger_j c_k c_l
 *
 * Has the symmetry properties: (see e.g. Sachdev's paper BH enrgopy and
 * strange metals)
 *
 * J_{ji;kl} = -J_{ij;kl}
 * J_{il;lk} = -J_{ij;kl}
 * J_{kl;ij} = J^*_{ij;kl}
 * < |J_{ij;kl}|^2 > = J^2
 *
 * This implies that for diagonal elements:
 * J_{ij;ij} is real and has variance \sigma^2 = J^2
 *
 * For off-diagonal elements J_{ij;kl} is complex and Re(J) and Im(J) have
 * variance \sigma^2 = J^2 / 2 .
 */
class KitaevDisorderParameter : public DisorderParameter {
public:
    // Create the object with given N and J parameters, and given
    // random number generator 'gen'.
    KitaevDisorderParameter(
            int N, double J, boost::random::mt19937* gen,
            bool complex_elements = true);

    virtual ~KitaevDisorderParameter();

    // Access J(i,j,k,l) (0-based)
    virtual cpx elem(int i, int j, int k, int l);

    boost::multi_array<cpx, 4> Jelems;
};

// J(i,j,k,l) = i-j+k-l
class MockDisorderParameter : public DisorderParameter {
public:
    virtual ~MockDisorderParameter();
    virtual cpx elem(int i, int j, int k, int l);
};

#endif // DISORDER_PARAMETER_H__
