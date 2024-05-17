#include <iostream>

typedef struct tensor {
    float* val;
    int* shape; // shape of the value tensor and gradient tensor
    float* grad;
    bool ignore; // ignore the tracking of this tensor?
    tensor (*fback)(tensor, tensor, tensor); // Pointer to the local backpropagation function
    tensor* parents; // tensors used to create this tensor
    int nb_parents; // nb of tensors used to create this tensor
} tensor;

int main() {
    std::cout << sizeof(struct tensor);
}
