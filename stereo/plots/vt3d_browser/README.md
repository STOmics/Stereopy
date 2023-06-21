# test_stereopy_3D_browser

## dependences

```
json
anndata>=0.8.0
numpy
pandas
```

## loading data
```
import anndata as ad
adata = ad.read_h5ad("D:/L3_b.h5ad")
```
## start a 3D atlas server without mesh
```
from stereopy_3D_browser import launch
import anndata as ad

adata = ad.read_h5ad("D:/L3_b.h5ad")

launch(adata,meshes={},cluster_label='annotation',spatial_label='spatial')
```

## start a 3D atlas server with meshes
```
from stereopy_3D_browser import launch
import anndata as ad

adata = ad.read_h5ad("D:/L3_b.h5ad")

launch(adata,meshes={'shell':'D:/shell.obj','midgut':'D:/midgut.obj'},cluster_label='annotation',spatial_label='spatial')
```

## start a 3D atlas server with meshes and with a list of anndata ( one slice per h5ad)
```
from stereopy_3D_browser import launch
import anndata as ad

datas = []
for i in range(16):
    datas.append(ad.read_h5ad(f"D:/Flysta3D/Summer/{i}.h5ad"))

launch(datas,meshes={'scene':'D:/Flysta3D/Summer/scene.obj'},cluster_label='annotation',spatial_label='spatial_rigid')
```
