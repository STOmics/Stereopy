SpaTrack
====================
Trajectory inference (TI) provides important insights in understanding cell development and biological process.
However, the integration of transcriptomic profiles and spatial locations to organize spatiotemporal cell orders is currently remaining challenges. 
Here we introduce spaTrack, which effectively constructs cell trajectories from an optimal-transport matrix at single cell resolution, 
taking into account both profile of gene expression and distance cost of cell transition in a spatial context.

spaTrack has the potential to capture fine local details of trajectory within a single tissue section of spatial transcriptomics (ST) data, 
as well as reconstruct cell dynamics across multiple tissue sections in a time series. To capture potential dynamic drivers, 
spaTrack models the fate of a cell as a function of expression profile along the time points driven by transcription factors, 
which facilitates the identification of key molecular regulators that govern cellular trajectories.

.. nbgallery::

    Apply_spaTrack_on_spatial_data_of_axolotl_brain_regeneration_after_injury
    Apply_spaTrack_on_spatial_data_of_Intrahepatic_cholangiocarcinoma_cancer
    Apply_spaTrack_to_infer_a_trajectory_on_spatial_transcriptomic_data_from_multiple_time_slices_of_axolotl_brain_regeneration
    Apply_spaTrack_to_infer_cell_transitions_across_multiple_time_points_in_spatial_transcriptomic_data_from_the_mouse_midbrain
    Apply_spaTrack_to_infer_cell_trajectory_in_scRNA-seq_data_from__hematopoietic_stem_cells_development_with_multiple_directions