# Issue #394: about generate_loom func

Hello. I am using the generate_loom method to generate loom, adata.var_names is gene name, there are duplicate gene names; but the matrix through st.io.read_gef and st.io.stereo_to_anndata, adata.var_names is gene id.

Can we obtain a Loom file that contains the gene id.
```
loom_data = generate_loom(
                gef_path=bgef_file,
                gtf_path=gtf_file,
                bin_type='bins',
                bin_size=50,
                out_dir=out_dir
                )
import scanpy as sc

loom_path = "./rna_velocity.loom"
adata = sc.read_loom(loom_path)
```