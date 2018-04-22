#ifndef __FACTORIZED_SPACE_UTILS_H__
#define __FACTORIZED_SPACE_UTILS_H__

#include "defs.h"
#include "eigen_utils.h"
#include "FockSpaceUtils.h"
#include "MajoranaDisorderParameter.h"

using namespace std;

typedef enum {
    EVEN_CHARGE = 0,
    ODD_CHARGE = 1
} ChargeParity;

// Hilbert space that's a tensor product of left/right spaces
class FactorizedSpace : public Space {
public:
    FactorizedSpace();
    FactorizedSpace(const FactorizedSpace& other);
    virtual FactorizedSpace& operator=(const FactorizedSpace& other);

    // N = number of Majorana fermions
    // Cuts the space in half (or close to half)
    static FactorizedSpace from_majorana(int N);

    // Put N_left Majorana's in the left space and N-N_left in the
    // right space
    static FactorizedSpace from_majorana(int N, int N_left);

    // Nd = number of Dirac fermions
    static FactorizedSpace from_dirac(int Nd);

    // Nd_left = number of Dirac fermions on left space
    static FactorizedSpace from_dirac(int Nd, int Nd_left);

    Space left;  // left factor
    Space right; // right factor

    // Shape of a single block (even/odd parity) of a state
    int state_block_rows();
    int state_block_cols();
    int state_block_size();

    // Size (no. elements) of a whole state (with 2 blocks)
    int state_size();

    // Storage size of a whole state
    size_t state_alloc_size();

    // Shape of a single block (even/odd parity) of a left operator
    int left_block_rows();
    int left_block_cols();
    int left_block_size();

    // Shape of a single block (even/odd parity) of a right operator
    int right_block_rows();
    int right_block_cols();
    int right_block_size();

 protected:
    void fact_init_dirac(int Nd);
    void fact_init_dirac(int Nd, int Nd_left);
};

// A factorized state with a given parity, embedded in the full
// factorized space. 
class FactorizedParityState {
public:
    // Initialize a zero state
    FactorizedParityState(FactorizedSpace& space,
                          ChargeParity charge_parity);

    // Initialize a random state with given parity
    FactorizedParityState(FactorizedSpace& space,
                          ChargeParity charge_parity,
                          boost::random::mt19937* gen);

    FactorizedParityState(const FactorizedParityState& other);
    FactorizedParityState& operator= (const FactorizedParityState& other);

    // Total number of elements (for fixed parity)
    int size();

    FactorizedSpace space;
    ChargeParity charge_parity;

    // The state is stored in the basis where the even Fock states
    // come first in each subspace. For even parity, the top-left and
    // bottom-right blocks are used. For odd parity, the top-right and
    // bottom-left blocks are used.
    Mat matrix;
};

// TODO replace uses of these functions with the class above

// Returns the given state in the factorized space matrix representation,
// with even-parity appearing before odd-parity states for each of the
// subspaces.
Mat get_factorized_state(const FactorizedSpace& space,
                         const Vec& state);

// The inverse of get_factorized_state
Vec get_unfactorized_state(const FactorizedSpace& space,
                           const Mat& factorized_state);

// Create a random state with no specific charge parity.
// If gen is null, uses eigen's random number generator.
Mat get_factorized_random_state(const FactorizedSpace& space,
                                boost::random::mt19937* gen);

// Create a random state with a given charge parity.
// If gen is null, uses eigen's random number generator.
Mat get_factorized_random_state(const FactorizedSpace& space,
                                ChargeParity charge_parity,
                                boost::random::mt19937* gen);

// Return a random unit vector with the given dimension
Vec get_random_state(const Space& space, boost::random::mt19937* gen);

#endif // __FACTORIZED_SPACE_UTILS_H__
