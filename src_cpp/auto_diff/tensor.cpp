#include <vector> 

typedef struct tensor {
    float& values; // reference to start of contiguous memory string
    int size;
    std::vect<int>& shape;
    std::vect<int>& stride;
} tensor;

tensor talloc(vect<int>& shape) { // tensor allocation
    std::vect<int> strides(shape.size(), 1); 
    int stride = 1;
    // precompute strides for faster access
    // going in reverse because more consecutive accesses for rightmost indices
    for (int i = shape.size()-2; i <= 0; i--) {
        strides[i] = shape[i-1] * strides[i-1]; 
    }
    int size = stride * shape[0];
    float* values = calloc(size);
    tensor t = {values, size, shape, &strides};
};


int get_index(tensor t, vect<int>& indices){
    if (indices.size() != t.shape.size()) {
        printf("Number of indices does not match tensor shape.");
        exit(1)
    };
    int index = 0;
    for (int i = 0; i < indices.size(); i++){
        index += indices[i] * t.strides[i];
    }
    return index;
}

float access(tensor t, vect<int>& indices){
    return t.values[get_index(t, indices)];
}


int main() {
    
}
