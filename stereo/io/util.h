
#ifndef GENETOH5_UTIL_H
#define GENETOH5_UTIL_H

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <climits>
#include <algorithm>
#include <condition_variable>
#include <thread>
#include <zlib.h>
#include <queue>
#include <cstring>
#include <cassert>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;

static constexpr char gene_merge[] = "/geneExp";
static constexpr char dnb_merge[] = "/wholeExp";
static constexpr unsigned int version = 2;
static constexpr size_t char_len = 32;

const int READLEN = 256*1024;

typedef struct 
{
    unsigned int x;
    unsigned int y;
    unsigned int cnt;
}Gene;

struct GenePos
{
    GenePos(const char* g, unsigned int o, unsigned c)
    {
        int i = 0;
        while (g[i] != '\0')
        {
            gene[i] = g[i];
            ++i;
        }
        offset = o;
        count = c;
    }
    char gene[char_len] = {0};
    unsigned int offset;
    unsigned int count;
};

struct GeneInfo
{
    GeneInfo(const char *ptr):geneid(ptr){};
    const char *geneid;
    std::vector<Gene> *vecptr;
};

struct Genedata
{
    Genedata(const char *ptr):geneid(ptr),umicnt(0){};
    const char *geneid;
    unsigned long umicnt;
    float e10;
    float c50;
    unsigned int maxexp;
    std::vector<Gene> *vecdataptr;
};

typedef struct
{
    unsigned short count;
    unsigned short genes;
}StatUC;

typedef struct
{
    unsigned int count;
    unsigned int genes;
}Stat;

typedef struct
{
    unsigned int min_x;
    unsigned int len_x;
    unsigned int min_y;
    unsigned int len_y;
    unsigned int max_mid;
    unsigned int max_gene;
    unsigned long number;
} DnbAttr;

typedef struct
{
    DnbAttr dnb_attr;
    StatUC *pmatrix_uc;
    Stat *pmatrix;
    //std::unordered_map<unsigned long, Stat> map_stat;
}DnbMatrix;

struct GeneErank
{
    GeneErank(const char *ptr):geneid(ptr){};
    const char *geneid;
    unsigned long umicnt;
    float e10;
    float c50;
    char attribute[10];
};

typedef struct
{
    unsigned int min_x;
    unsigned int min_y;
    unsigned int max_x;
    unsigned int max_y;
    unsigned int max_exp;
} ExpAttr;

struct GeneStat
{
    GeneStat(std::string& g, unsigned int m, float e)
    {
        memcpy(gene, g.c_str(), g.size());
        MIDcount = m;
        E10 = e;
    }
    char gene[char_len] = {0};
    unsigned int MIDcount;
    float E10;
};

struct CommonPara
{
    unsigned int resolution;
};

#endif 