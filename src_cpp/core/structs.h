typedef struct tensor {
    float* values; // reference to start of contiguous memory string
    int size;
    std::vector<int> shape;
    bool permute; // wether or not the axes where permuted
    std::vector<int> permutes; // normal: <0, 1, 2> modified ex.: <1,2,0>
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
