SpaTrack
====================

Introducion
--------------------

Trajectory inference (TI) provides important insights in understanding cell development and biological process.
However, the integration of transcriptomic profiles and spatial locations to organize spatiotemporal cell orders is currently remaining challenges. 
Here we introduce spaTrack, which effectively constructs cell trajectories from an optimal-transport matrix at single cell resolution, 
taking into account both profile of gene expression and distance cost of cell transition in a spatial context.

spaTrack has the potential to capture fine local details of trajectory within a single tissue section of spatial transcriptomics (ST) data, 
as well as reconstruct cell dynamics across multiple tissue sections in a time series. To capture potential dynamic drivers, 
spaTrack models the fate of a cell as a function of expression profile along the time points driven by transcription factors, 
which facilitates the identification of key molecular regulators that govern cellular trajectories [`Shen <https://www.biorxiv.org/content/10.1101/2023.09.04.556175v2>`_].

Highlighted features
---------------------

1. reconstructs fine local trajectories from ST data.
2. integrates spatial transition matrix of multiple samples to generate complete trajectories.
3. traces cell trajectory across multiple tissue sections via direct mapping without integrating data.
4. captures the driven factors of differentiation.
5. could be extensively applied on both ST data and scRNA-seq data.
6. requires lower computing memory and loads than RNA-velocity methods, making it a fast and effective option for TI study.

Preparation
---------------------

Torch is the necessary dependency and needs to be installed first.

    pip install pysal==2.6.0 pygam==0.8.0

    CPU: pip install torch==2.4.1+cpu --extra-index-url https://download.pytorch.org/whl

    GPU(CUDA11): pip install torch==2.4.1+cu118 --extra-index-url https://download.pytorch.org/whl/

    GPU(CUDA12): pip install torch==2.4.1+cu124 --extra-index-url https://download.pytorch.org/whl/

Turorials
---------------------

We provide the following five tutorials as reference cases to illustrate the application of spaTrack in inferring cell trajectories 
on ST data of single slices with a single starting point and multiple starting points, as well as ST data of multiple time slices, and scRNA-seq data.

.. nbgallery::

    Apply_spaTrack_on_spatial_data_of_axolotl_brain_regeneration_after_injury
    Apply_spaTrack_on_spatial_data_of_Intrahepatic_cholangiocarcinoma_cancer
    Apply_spaTrack_to_infer_a_trajectory_on_spatial_transcriptomic_data_from_multiple_time_slices_of_axolotl_brain_regeneration
    Apply_spaTrack_to_infer_cell_transitions_across_multiple_time_points_in_spatial_transcriptomic_data_from_the_mouse_midbrain
    Apply_spaTrack_to_infer_cell_trajectory_in_scRNA-seq_data_from__hematopoietic_stem_cells_development_with_multiple_directions