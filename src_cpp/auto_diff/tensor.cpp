#include <iostream>

struct tensor {
    float* val;
    int* shape; // shape of the value tensor and gradient tensor
    float* grad;
    bool ignore; // ignore the tracking of this tensor?
    struct tensor (*fback)(struct tensor, struct tensor, bool track); // Pointer to the local backpropagation function
    struct tensor* parents; // tensors used to create this tensor
    int nb_parents; // nb of tensors used to create this tensor
};

int main() {
    std::cout << sizeof(struct tensor);
}
