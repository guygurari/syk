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
    }

    return 0;
}
