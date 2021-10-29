#include "gef_read.h"

#include <ctime>

//vector<unsigned long long> cpp_uniq_cell_index(const vector<unsigned int> &vec_x, const vector<unsigned int> &vec_y, vector<unsigned int> &rows) {
int cpp_uniq_cell_index(const unsigned int *vec_x, const unsigned int *vec_y, unsigned int *rows, unsigned long long len){
    time_t now = time(0);
    char* dt = ctime(&now);
    printf("cpp_uniq_cell_index start = %s\n", dt);


//    unsigned long long xy_id;
//    vector<unsigned long long> cells;
      int cells = 0;
//    map<unsigned long long, long> cells_map;
//    auto first = vec_x.begin();
//    auto end = vec_x.end();
//    auto first_y = vec_y.begin();
//    long index = 0;
//
//    while (first != end)
//    {
//        xy_id = *first;
//        xy_id = xy_id << 32 | *first_y;
//        auto iter = cells_map.find(xy_id);
//
//        if(iter != cells_map.end()){
//            rows.push_back(iter->second);
//        }else {
//            rows.push_back(index);
//            cells.push_back(xy_id);
//            cells_map.insert(map<unsigned long long, long>::value_type (xy_id, index));
//            ++index;
//        }
//
//        ++first;
//        ++first_y;
//    }
//    now = time(0);
//    dt = ctime(&now);
    printf("cpp_uniq_cell_index end = %s\n", dt);
    return cells;
}


int cpp_gene_count_index(const vector<int> &gene_count, vector<int> &cols) {
    int gene_index = 0;
    for(int c : gene_count){
        for (int i = 0; i < c; ++i) {
            cols.push_back(gene_index);
        }
        gene_index ++;
    }
    return gene_index;
}
