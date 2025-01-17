#include <vector> 
#include <cstdlib>
#include <iostream>
#include "structs.h"
tensor matmul(tensor x,tensor y);
tensor talloc(std::vector<int> shape);
int get_index(tensor t, std::vector<int> indices);
float access(tensor t, std::vector<int> indices);
void set(tensor t, float value, std::vector<int> indices);
int min(int a, int b);
void fill_diagonal(tensor t, float value);
void print(tensor t);
tensor sum(tensor x, tensor y);
tensor matmul(tensor x, tensor y);
tensor hori_range(std::vector<int> indices);
tensor identity(std::vector<int> indices);
void permute(std::vector<int>& vec, std::vector<permutation>);
tensor permute(tensor t, std::vector<int> permutation);
tensor permute(tensor t, int axis_a, int axis_b);
tensor merge(tensor t, int axis_a, int axis_b);
tensor split(tensor t, int axis, std::vector<int> splitting);