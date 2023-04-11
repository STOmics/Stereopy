import os

if os.name == 'nt':
    TEST_IMAGE_PATH = os.path.dirname(__file__) + '\\test_image\\'
    os.makedirs(TEST_IMAGE_PATH, exist_ok=True)
    TEST_DATA_PATH = os.path.dirname(__file__) + '\\test_data\\'
else:
    # os.name == 'posix' or os.name == 'mac' and other OSs
    TEST_IMAGE_PATH = os.path.dirname(__file__) + '/test_image/'
    os.makedirs(TEST_IMAGE_PATH, exist_ok=True)
    TEST_DATA_PATH = os.path.dirname(__file__) + '/test_data/'

# SS200000132BR_A1.bin1.Lasso.gem.gz
DEMO_132BR_A1_URL = 'https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?' \
                    'shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&' \
                    'nodeId=8a80804a837dc46f01838302491a21e0&code='

# SS200000132BR_A2.bin1.Lasso.gem.gz
DEMO_132BR_A2_URL = 'https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?' \
                    'shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&' \
                    'nodeId=8a80804a837dc46f01838302df5b21e2&code='

# SS200000135TL_D1.tissue.gef
DEMO_DATA_URL = 'https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?' \
                'shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&' \
                'nodeId=8a80804a837dc46f018382c40ca51af0&code='

# SS200000135TL_D1.tissue.gem.gz
DEMO_DATA_135_TISSUE_GEM_GZ_URL = 'https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?' \
                                  'shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&' \
                                  'nodeId=8a808043837dc3ac0183d578b9a772e0&code='

# SS200000141TL_B5_raw.h5ad
DEMO_H5AD_URL = 'https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?' \
                'shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&' \
                'nodeId=8a80804a837dc46f018387fbe67c63c7&code='

# TODO
# GSE84133_GSM2230761_mouse1.anndata075.h5ad
DEMO_REF_URL = 'https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?' \
               'shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&' \
               'nodeId=8a80804386ed81950187315defb43bc2&code='

# TODO
# GSE84133_GSM2230762_mouse2.anndata075.h5ad
DEMO_TEST_URL = 'https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?' \
                'shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&' \
                'nodeId=8a80804386ed81950187315e260e3bc7&code='

# test_mm_mgi_tfs.txt
DEMO_TFS_URL = 'https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?' \
                    'shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&' \
                    'nodeId=8a80804386ed8195018725dae9ef3c24&code='

# mm10_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather
DEMO_DATABASE_URL = 'https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?'\
                        'shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&' \
                        'nodeId=8a80804386ed8195018725dac1363c20&code='

# motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl
DEMO_MOTIF_URL = 'https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?' \
                    'shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&' \
                    'nodeId=8a80804386ed8195018725dae91a3c22&code='

