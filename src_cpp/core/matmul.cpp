#include "utero.h"

tensor matmul(tensor x, tensor y) {
    if ( x.shape[1] != y.shape[0] ) {
        printf("Number of columns must match number of rows.");
        exit(1);
    };

    // compute matrix multiplication in naive O(mnr) time
    tensor z = talloc({x.shape[0], y.shape[1]});
    for (int i = 0; i < x.shape[0]; i++){
        for (int j = 0; j < y.shape[1]; j++) {
            float s = 0;
            for (int k = 0; k < x.shape[1]; k++){
                s += access(x, {i, k}) * access(y, {k, j});
            }
            set(z, s, {i,j});
        }
    }

    return z;
};

var matmul(var x, var y) {
    // TODO
}

