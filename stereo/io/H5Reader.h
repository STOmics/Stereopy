
#pragma once

#include <string>
#include <vector>
#include <iostream>
#include "hdf5.h"

using namespace std;

class H5Reader
{
public:
    H5Reader(const string &filename, int bin_size);

    ~H5Reader();
    //expressions number
    int bin_size = 0;
    unsigned int minX{}, minY{}, maxX{}, maxY{}, gene_num{}, cell_num{};
    unsigned long long exp_len = 0;

    unsigned long long getExpLen() const;
    unsigned int getGeneNum() const;

    vector<unsigned long long> getExpData(unsigned int * cell_index, unsigned int * count);

    void getGeneData(unsigned int * gene_index, vector<string> & uniq_genes);

private:
    hid_t m_file_id;
    herr_t m_status{};
    // hid_t m_group_id;
    hid_t exp_dataspace_id{};
    hid_t exp_dataset_id{};
    hid_t gene_dataspace_id{};
    hid_t gene_dataset_id{};

    void openExpressionSpace();
    void openGeneSpace();
};
