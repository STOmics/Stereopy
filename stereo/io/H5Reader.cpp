#include "H5Reader.h"
#include "util.h"
#include "khash.h"

#include <iostream>
#include <cstdio>

using namespace std;

KHASH_MAP_INIT_INT64(m64, unsigned int)

H5Reader::H5Reader(const string& filename, int bin_size)
{
    this->m_file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    this->bin_size = bin_size;
    openExpressionSpace();
    openGeneSpace();
}

H5Reader::~H5Reader()
{
    H5Fclose(m_file_id);
    H5Dclose(exp_dataset_id);
    H5Sclose(exp_dataspace_id);
    H5Dclose(gene_dataset_id);
    H5Sclose(gene_dataspace_id);
}

vector<unsigned long long> H5Reader::getExpData(unsigned int * cell_index, unsigned int * count)
{
//    time_t now = time(nullptr);
//    char* dt = ctime(&now);
//    printf("cpp_uniq_cell_index start = %s\n", dt);

    hid_t memtype;
    hid_t attr;
    unsigned long long uniq_cell_id;

    memtype = H5Tcreate(H5T_COMPOUND, sizeof(Gene));
    m_status = H5Tinsert(memtype, "x", HOFFSET(Gene, x), H5T_NATIVE_UINT);
    m_status = H5Tinsert(memtype, "y", HOFFSET(Gene, y), H5T_NATIVE_UINT);
    m_status = H5Tinsert(memtype, "count", HOFFSET(Gene, cnt), H5T_NATIVE_UINT);

    Gene* expData;
    expData = (Gene*)malloc(exp_len * sizeof(Gene));
    m_status = H5Dread(exp_dataset_id, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, expData);

//    now = time(nullptr);
//    dt = ctime(&now);
//    printf("* H5Dread end = %s\n", dt);

//    map<unsigned long long, unsigned int> cells_map;
//    unordered_map<unsigned long long, unsigned int> cells_map;
    vector<unsigned long long> uniq_cells;
    unsigned int index = 0;

    int absent, is_missing;
    khint_t k;
    khash_t(m64) *h = kh_init(m64);  // allocate a hash table

    for (int i = 0; i < exp_len; ++i) {
        uniq_cell_id = expData[i].x;
        uniq_cell_id = uniq_cell_id << 32 | expData[i].y;

        k = kh_get(m64, h, uniq_cell_id);
        is_missing = (k == kh_end(h));
        if (!is_missing){
            cell_index[i] = kh_value(h, k);
        }else {
            cell_index[i] = index;
            uniq_cells.push_back(uniq_cell_id);
            k = kh_put(m64, h, uniq_cell_id, &absent);  // insert a key to the hash table
            kh_value(h, k) = index;
            ++index;
        }

//        auto iter = cells_map.find(uniq_cell_id);
//        if(iter != cells_map.end()){
//            cell_index.push_back(iter->second);
//        }else {
//            cell_index.push_back(index);
//            uniq_cells.push_back(uniq_cell_id);
//            cells_map.insert(unordered_map<unsigned long long, unsigned int>::value_type (uniq_cell_id, index));
//            ++index;
//        }
//
        count[i] = expData[i].cnt;
    }

    this->cell_num = index;
//    now = time(nullptr);
//    dt = ctime(&now);
//    printf("index = %d\n", index);
//    printf("* cell_index end = %s\n", dt);

    // Read attribute of raw data
    attr = H5Aopen(exp_dataset_id, "minX", H5P_DEFAULT);
    // m_dataspace_id = H5Aget_space(attr);
    // ndims = H5Sget_simple_extent_dims(space, dims, NULL);
    m_status = H5Aread(attr, H5T_NATIVE_UINT, &minX);
    attr = H5Aopen(exp_dataset_id, "minY", H5P_DEFAULT);
    m_status = H5Aread(attr, H5T_NATIVE_UINT, &minY);
    attr = H5Aopen(exp_dataset_id, "maxX", H5P_DEFAULT);
    m_status = H5Aread(attr, H5T_NATIVE_UINT, &maxX);
    attr = H5Aopen(exp_dataset_id, "maxY", H5P_DEFAULT);
    m_status = H5Aread(attr, H5T_NATIVE_UINT, &maxY);

    if (expData != nullptr)
        free(expData);

    H5Aclose(attr);
    kh_destroy(m64, h);
    H5Tclose(memtype);
    return uniq_cells;
}

void H5Reader::getGeneData(unsigned int * gene_index, vector<string> & uniq_genes)
{
    hid_t memtype;
    hid_t strtype;

    strtype = H5Tcopy(H5T_C_S1);
    m_status = H5Tset_size(strtype, char_len);

    memtype = H5Tcreate(H5T_COMPOUND, sizeof(GenePos));
    m_status = H5Tinsert(memtype, "gene", HOFFSET(GenePos, gene), strtype);
    m_status = H5Tinsert(memtype, "offset", HOFFSET(GenePos, offset), H5T_NATIVE_UINT);
    m_status = H5Tinsert(memtype, "count", HOFFSET(GenePos, count), H5T_NATIVE_UINT);

    GenePos* idxData;
    // cout<<"genes number: "<<idxLen<<endl;
    idxData = (GenePos*)malloc(gene_num * sizeof(GenePos));
    m_status = H5Dread(gene_dataset_id, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, idxData);

    // Write data line by line
    unsigned long long exp_len_index = 0;
    for (unsigned int i = 0; i < gene_num; ++i)
    {
        const char* gene = idxData[i].gene;
        uniq_genes.emplace_back(gene);
        unsigned int c = idxData[i].count;
        for (int j = 0; j < c; ++j)
        {
            gene_index[exp_len_index++] = i;
        }
    }

    assert(exp_len_index == exp_len);

    if (idxData != nullptr)
        free(idxData);

    H5Tclose(strtype);
    H5Tclose(memtype);
}

unsigned long long int H5Reader::getExpLen() const {
    return exp_len;
}

unsigned int H5Reader::getGeneNum() const {
    return gene_num;
}

void H5Reader::openExpressionSpace() {
    hsize_t dims[1];
    // Read raw data
    char expName[128]={0};
    sprintf(expName, "%s/bin%d/expression", gene_merge, bin_size);
    exp_dataset_id = H5Dopen(m_file_id, expName, H5P_DEFAULT);
    if (exp_dataset_id < 0)
    {
        cerr<<"failed open dataset: "<<expName<<endl;
        return;
    }
    exp_dataspace_id = H5Dget_space(exp_dataset_id);
    H5Sget_simple_extent_dims(exp_dataspace_id, dims, NULL);
    exp_len = dims[0];
}

void H5Reader::openGeneSpace() {
    hsize_t dims[1];

    // Read index
    char idxName[128]={0};
    sprintf(idxName, "%s/bin%d/gene", gene_merge, bin_size);
    gene_dataset_id = H5Dopen(m_file_id, idxName, H5P_DEFAULT);
    if (gene_dataset_id < 0)
    {
        cerr<<"failed open dataset: "<<idxName<<endl;
        return;
    }
    gene_dataspace_id = H5Dget_space(gene_dataset_id);
    H5Sget_simple_extent_dims(gene_dataspace_id, dims, NULL);
    gene_num = dims[0];
}
