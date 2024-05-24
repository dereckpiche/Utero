typedef struct tensor {
    float* values; // reference to start of contiguous memory string
    int size;
    std::vector<int> shape;
    std::vector<int> strides;
} tensor;

typedef struct var {
    tensor* val;
    tensor* grad;
    bool ignore; // ignore the tracking of this tensor?
    var (*fback)(var**, var); // Pointer to the local backpropagation function
    tensor** parents; // tensors used to create this tensor
    int nb_parents; // nb of tensors used to create this tensor
} var;
