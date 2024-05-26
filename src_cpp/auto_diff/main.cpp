#include "utero.h"

void sum_test(){
    tensor x = talloc({5,5});
    tensor y = talloc({5,5});
    fill_diagonal(x, 1.0);
    fill_diagonal(y, 1.0);
    tensor z = sum(x, y);
    print(z);
};

void matmul_test(){
    tensor x = identity({5,5});
    print(x);
    tensor y = hori_range({5,5});
    print(y);
    tensor z = matmul(x, y);
    print(z);
};

int main() {
    //sum_test();
    matmul_test();
}
