#include "utero.h"

tensor talloc(std::vector<int> shape) { // tensor allocation
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

int get_index(tensor t, std::vector<int> indices){
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

bool is_matrix(tensor t){
    if (t.shape.size()==2) return true;
    return false;
}

tensor hori_range(std::vector<int> indices){
    if (indices.size() !=2 ) {
        std::cout << "Must be a matrix shape.";
        exit(1);
    }
    tensor t = talloc(indices);
    for (int i = 0; i<indices[0]; i++){
        for (int j = 0; j<indices[1]; j++){
            set(t, (float)i, {i,j});
        }
    }
    return t;
}   

tensor identity(std::vector<int> indices){
    if (indices.size() !=2 ) {
        std::cout << "Must be a matrix shape.";
        exit(1);
    }
    tensor t = talloc(indices);
    fill_diagonal(t, 1.0);
    return t;
} 

float access(tensor t, std::vector<int> indices){
    return t.values[get_index(t, indices)];
}

void set(tensor t, float value, std::vector<int> indices){
    t.values[get_index(t, indices)] = value;
}

int min(int a, int b){
    if (a<b) return a;
    return b;
}

void fill_diagonal(tensor t, float value){
    if (t.shape.size()==2) {
        for (int i=0; i < min(t.shape[0], t.shape[1]); i++){
            set(t, value, {i,i});
        }
    }   
}

void print(tensor t){
    if (t.shape.size() == 2) {
        std::string matrix = "\n";
        for (int i=0; i< t.shape[0]; i++){
            for (int j=0; j<t.shape[1];j++){
                matrix += std::to_string(access(t, {i,j})) + " ";
            }
            matrix += "\n";
        }
        std::cout << matrix;
    }
}
 

