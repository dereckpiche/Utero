typedef struct tensor {
    float* values; // reference to start of contiguous memory string
    int size;
    std::vector<int> shape;
    std::vector<int> strides;
} tensor;

typedef struct var {
    tensor val;
    tensor grad;
    bool ignore; // ignore the tracking of this tensor?
    tensor (*chains)(tensor, tensor); // array of (grad, parent_value) -> parent_grad functions
    tensor* parents; // parents of this variable
    int nb_parents;
} var;
