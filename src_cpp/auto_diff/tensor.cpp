#include <vector> 
#include <cstdlib>
#include <iostream>

typedef struct tensor {
    float* values; // reference to start of contiguous memory string
    int size;
    std::vector<int> shape;
    std::vector<int> strides;
} tensor;

tensor talloc(std::vector<int> shape) { // tensor allocation
    std::cout << shape.size();
    std::vector<int> strides(shape.size(), 1); 
    // precompute strides for faster access
    // going in reverse because more consecutive accesses for rightmost indices
    for (int i = shape.size()-2; i >= 0; i--) {
        strides[i] = shape[i+1] * strides[i+1]; 
    }
    int size = strides[0] * shape[0];
    float* values = (float*)std::calloc(size, sizeof(float));
    tensor t = {values, size, shape, strides};
    return t;
}

int get_index(tensor t, std::vector<int>& indices){
    if (indices.size() != t.shape.size()) {
        printf("Number of indices does not match tensor shape.");
        exit(1);
    };
    int index = 0;
    for (int i = 0; i < indices.size(); i++){
        index += indices[i] * t.strides[i];
    }
    return index;
}

float access(tensor t, std::vector<int> indices){
    return t.values[get_index(t, indices)];
}

void set(tensor t, float value, std::vector<int> indices){
    // TODO;
}

void print(tensor t){
    std::cout << t.shape.size();
    if (t.shape.size() == 2) {
        std::cout << "yo";
        std::string matrix = "\n";
        for (int i=0; i< t.shape[0]; i++){
            for (int j=0; j<t.shape[1];j++){
                matrix += std::to_string(access(t, {i,j})) + " ";
            }
            matrix = matrix + "\n";
        }
        std::cout << matrix;
    }
}
 
int main() {
    tensor t = talloc({7,4});
    print(t);
    return 0;
}

