import os

if os.name == 'nt':
    TEST_IMAGE_PATH = os.path.dirname(__file__) + '\\test_image\\'
    os.makedirs(TEST_IMAGE_PATH, exist_ok=True)
    TEST_DATA_PATH = os.path.dirname(__file__) + '\\test_data\\'
    os.makedirs(TEST_DATA_PATH, exist_ok=True)
else:
    # os.name == 'posix' or os.name == 'mac' and other OSs
    TEST_IMAGE_PATH = os.path.dirname(__file__) + '/test_image/'
    os.makedirs(TEST_IMAGE_PATH, exist_ok=True)
    TEST_DATA_PATH = os.path.dirname(__file__) + '/test_data/'
    os.makedirs(TEST_DATA_PATH, exist_ok=True)

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

# GSE84133_GSM2230761_mouse1.anndata075.h5ad
DEMO_REF_URL = 'https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?' \
               'shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&' \
               'nodeId=8a80804a86ed7f8501876ee6aac7228b&code='

# GSE84133_GSM2230762_mouse2.anndata075.h5ad
DEMO_TEST_URL = 'https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?' \
                'shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&' \
                'nodeId=8a80804a86ed7f8501876ee6b905228d&code='

# test_mm_mgi_tfs.txt
DEMO_TFS_URL = 'https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?' \
               'shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&' \
               'nodeId=8a80804386ed8195018725dae9ef3c24&code='

# mm10_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather
DEMO_DATABASE_URL = 'https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?' \
                    'shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&' \
                    'nodeId=8a80804386ed8195018725dac1363c20&code='

# motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl
DEMO_MOTIF_URL = 'https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?' \
                 'shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&' \
                 'nodeId=8a80804386ed8195018725dae91a3c22&code='

# gene.gtf
DEMO_GTF_URL = 'https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?' \
               'shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&' \
               'nodeId=8a80804a837dc46f018382c713661b63&code='

# SS200000135TL_D1.cellbin.gef
DEMO_135_CELL_BIN_GEF_URL = "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?" \
                            "shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&" \
                            "nodeId=8a80804a837dc46f018382c2da791aec&code="

# SS200000135TL_D1.cellbin.gem
DEMO_135_CELL_BIN_GEM_URL = "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?" \
                            "shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&" \
                            "nodeId=8a808043837dc3ac0183d57a731b7331&code="

# 20210426-T173-Z3-L-M019-01_regist_21635_18385_9064_13184.tif
DEMO_CELL_SEGMENTATION_TIF_URL = "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?" \
                                 "shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&" \
                                 "nodeId=8a808043837dc3ac01840d148e2b1a31&code="

# cell_segmetation_v3.0.onnx
DEMO_CELL_SEGMENTATION_V3_MODEL_URL = "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?" \
                                      "shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&" \
                                      "nodeId=8a80804a86ed7f8501879db1c1b84496&code="

# demo 3d slice 0-15 AnnData 0.8.0
DEMO_3D_SLICE_0_15_URLS_LIST = [
    "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&nodeId=8a80804a86ed7f85018769d7dfda0317&code=",
    "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&nodeId=8a80804a86ed7f85018769d7eae9031b&code=",
    "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&nodeId=8a80804a86ed7f85018769d7f792031d&code=",
    "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&nodeId=8a80804a86ed7f85018769d80616031f&code=",
    "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&nodeId=8a80804a86ed7f85018769d814ff0321&code=",
    "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&nodeId=8a80804a86ed7f85018769d8242e0325&code=",
    "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&nodeId=8a80804a86ed7f85018769d835c30327&code=",
    "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&nodeId=8a80804a86ed7f85018769d84e430329&code=",
    "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&nodeId=8a80804a86ed7f85018769d866e4032b&code=",
    "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&nodeId=8a80804a86ed7f85018769d87ebf032d&code=",
    "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&nodeId=8a80804a86ed7f85018769d893a8032f&code=",
    "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&nodeId=8a80804a86ed7f85018769d8ab160331&code=",
    "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&nodeId=8a80804a86ed7f85018769d8be880333&code=",
    "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&nodeId=8a80804a86ed7f85018769d8d3530335&code=",
    "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&nodeId=8a80804a86ed7f85018769d8eb4b0337&code=",
    "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&nodeId=8a80804a86ed7f85018769d90078033b&code="
]

# mouse_forebrain.anndata_075.h5ad
DEMO_FORE_BRAIN_DATA_URL = "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?" \
                           "shareEventId=share_2022928142945896_010df2aa7d344d97a610557de7bad81b&" \
                           "nodeId=8a8080438842c12f0188522965a23618&code="
