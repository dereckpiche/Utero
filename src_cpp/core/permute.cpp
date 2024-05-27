void permute(std::vector<int>& vec, std::vector<permutation>){
    std::vector<int> temp(vec);
    for (int i = 0; i<permutation.size(); i++){
        vec[i] = temp[permutation[i]];
    }   
}

tensor permute(tensor t, std::vector<int> permutation){
    permute(t.shape);
    permute(t.strides);
    return t;
}

tensor permute(tensor t, int axis_a, int axis_b){
    int temp = t.shape[axis_a];
    t.shape[axis_a] = t.shape[axis_b];
    t.shape[axis_b] = temp;
    temp = t.strides[axis_a];
    t.strides[axis_a] = t.shape[axis_b];
    t.strides[axis_b] = temp;
    return t;
}