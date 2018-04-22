////////////////////////////////////////////////////////////
//
// Print check
// 
// 
////////////////////////////////////////////////////////////

#include <iostream>
#include <iomanip>
#include <fstream>
#include "CudaLanczos.h"

int main(int argc, char *argv[]) {
    argc--; argv++;
    cout << setprecision(precision);

    if (argc == 0) {
        cout << "Usage: lanczos-checkpoint-info lanc-N40-run1-state ...\n\n";
        return 1;
    }

    for (int i = 0; i < argc; i++) {
        string filename(argv[i]);
        int num_steps;
        int max_steps;
        if (!CudaLanczos::get_state_info(filename, num_steps, max_steps)) {
            cerr << "Cannot read state from file " << filename << endl;
            return 1;
        }

        cout << filename << ":  " << num_steps << " of " << max_steps
             << " steps\n";
    }

    return 0;
}
