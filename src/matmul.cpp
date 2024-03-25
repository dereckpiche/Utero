#include <vector>
#include <cassert>


class MatMul : public Op {
public:
    void forward(tensor A, tensor B) override {
        // Simplified for clarity: actual implementation should consider matrix dimensions
        int rows = inputs.size(); // Number of rows in the first matrix (A)
        int cols = inputs[0].size(); // Number of columns in the second matrix (B)
        outputs.resize(rows, std::vector<float>(cols, 0)); // Resize output matrix

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                for (int k = 0; k < inputs[0].size(); ++k) { // Assuming square matrices for simplicity
                    outputs[i][j] += inputs[0][i][k] * inputs[1][k][j];
                }
            }
        }
    }

    void backward(tensor grad_z) override {
        // Simplified gradient computation for matrix multiplication
        // Actual implementation should compute gradients w.r.t. inputs considering matrix dimensions
        gradients.resize(inputs.size(), std::vector<std::vector<float>>(inputs[0].size(), std::vector<float>(inputs[0][0].size(), 0)));

        // Gradient of matrix multiplication w.r.t. the first matrix (A)
        for (int i = 0; i < inputs[0].size(); ++i) {
            for (int j = 0; j < inputs[1][0].size(); ++j) {
                for (int k = 0; k < inputs[1].size(); ++k) {
                    gradients[0][i][k] += gradients[0][i][j] * inputs[1][k][j];
                }
            }
        }

        // Gradient of matrix multiplication w.r.t. the second matrix (B)
        for (int i = 0; i < inputs[0].size(); ++i) {
            for (int j = 0; j < inputs[1][0].size(); ++j) {
                for (int k = 0; k < inputs[0][0].size(); ++k) {
                    gradients[1][k][j] += inputs[0][i][k] * gradients[0][i][j];
                }
            }
        }
    }
};