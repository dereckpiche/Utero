#include <vector> 

typedef struct vect {
    float* emts; // pointer to first element in the memory
    int size; // total number of elements
}

typedef struct tensor {
    float* values;
    std::vect<int>& shape;
    std::vect<int>& stride;
} tensor;


tensor talloc(const vect<int>& shape) {
    float* values = malloc(shape.size());
    for (int i = 0; i < shape.size(); i++) {
        
    }
};


int get_index(tensor t, const vect<int>& indices){
    if (indices.size() != t.shape.size()) {
        printf("Number of indices does not match.");
        exit(1)
    };
    int index = 0;
    for (int i = 0; i < indices.size(); i++){
        index+=indices[i]*t.strides[i];
    }
    return index;
}

