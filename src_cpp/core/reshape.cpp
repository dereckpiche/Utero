#include <utero.h>

tensor merge(tensor t, int axis_a, int axis_b){
    // TODO: catch error, verify
    // axis a goes into axis b
    t.shape[axis_b] *= t.shape[axis_a];
    t.shape.erase(axis_a)
    t.strides = get_strides(t.shape);
    return t;
}
tensor split(tensor t, int axis, std::vector<int> splitting){
    // TODO: catch error, verify
    for (int i = 0, i < splitting.size(); i++){
        t.shape.emplace([axis+i], splitting[i]);
    }
    t.strides = get_strides(t.shape);
}