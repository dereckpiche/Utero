#include "utero.h"



void sum_test(){
    tensor x = talloc({5,5});
    tensor y = tallco({5,5});
    fill_diagonnal(x, 1.0);
    fill_diagonnal(y, 1.0);
    tensor z = sum(x, y);
    print(z);
};

int main() {
    sum_test();
}
