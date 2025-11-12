# test_stereopy_3D_browser

## Installation

**As an under-developing project, no pip or conda installation supports for now.**

```
git clone https://github.com/cchd0001/test_stereopy_3D_browser.git  your-local-folder
```

## dependences

Except the python standard libraries, we also reley on below packages:

```
anndata>=0.8.0
numpy
pandas
```

Try the below codes:

```
sys.path.append('your-local-folder')
from stereopy_3D_browser import launch,endServer
```

## Usage

### Quick start for explore annotation and gene expression data

```
# import our code 
import sys
sys.path.append('your-local-folder')
from stereopy_3D_browser import launch,endServer

# import annconda and load input data
import anndata as ad
adata = ad.read_h5ad("your-target-h5ad")

# launch the browser now 
launch(adata,meshes={},cluster_label=['annotation'] ,spatial_label='spatial',port=7654)

# now please interactive browse your data
# if your need shutdown the server, run 
endServer("127.0.0.1",7654) 
```

### More examples:

see the examples/xxx.ipynb for examples of mesh/GRN/CCC/PAGA functions.