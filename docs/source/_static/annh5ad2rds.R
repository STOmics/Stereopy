library(SeuratDisk)
library(Seurat)
library(ggplot2)
library(dplyr)
library(rjson)
library(argparser)
args <- arg_parser("Converting h5ad file(.h5ad) to RDS.")
args <- add_argument(args, "--infile", help = "input .h5ad file")
args <- add_argument(args, "--outfile", help = "output RDS file")
argv <- parse_args(args)

infile <- argv$infile
outfile <- argv$outfile

print("Converting h5ad to h5seurat file.")
Convert(infile, dest = "h5seurat",assay = "Spatial", overwrite = TRUE)
h5file<-paste(paste(unlist(strsplit(infile,"h5ad",fixed=TRUE)),collapse='h5ad'),"h5seurat",sep="")
print("Loading h5seurat file.")
Stdata <- LoadH5Seurat(h5file)

if(!is.null(Stdata@reductions$ignore)){
	Stdata@reductions$ignore <- NULL
}
if(!is.null(Stdata@reductions$spatial)){
	Stdata@reductions$spatial <- NULL
}
if(!is.null(Stdata@misc$raw_counts)){
	if(!is.null(Stdata@misc$raw_genename)){
		Stdata@misc$raw_counts@Dimnames[[1]]  = Stdata@misc$raw_genename
		Stdata@misc$raw_genename<-NULL
	}
	if(!is.null(Stdata@misc$raw_cellname)){
		Stdata@misc$raw_counts@Dimnames[[2]]  = Stdata@misc$raw_cellname
		Stdata@misc$raw_cellname<-NULL
	}
	print("Adding misc raw_counts to Spatial assay as counts.")
	Stdata@assays$Spatial@counts=Stdata@misc$raw_counts
	if(dim(Stdata[['Spatial']]@scale.data)[1]>0){
		print("Adding misc raw_counts to Spatial assay as data.")
		Stdata@assays$Spatial@data=Stdata@misc$raw_counts
	}
	Stdata@misc$raw_counts<-NULL
}
if(!is.null(Stdata@misc$sct_data)){
	print("Creating SCT assay.")
	Stdata@assays$SCT=Stdata@assays$Spatial
	Stdata@assays$SCT@data=Stdata@misc$sct_data
	Stdata@misc$sct_data<-NULL
	Stdata@assays$Spatial@scale.data=matrix(,nrow=10,ncol=0)

	f_len<-length(Stdata@assays$SCT@meta.features)
	if(f_len>0){
		Stdata[['SCT']]@meta.features <- Stdata[['SCT']]@meta.features[,-(1:f_len)]
	}
	if(!is.null(Stdata@misc$sct_genename)){
		print("Adding gene dimnames to SCT data.")
		Stdata@assays$SCT@data@Dimnames[[1]]  = Stdata@misc$sct_genename
	}
	if(!is.null(Stdata@misc$sct_cellname)){
		print("Adding cell dimnames to SCT data.")
		Stdata@assays$SCT@data@Dimnames[[2]]  = Stdata@misc$sct_cellname
	}
	#Stdata@assays$SCT@data@Dimnames=Stdata@assays$Spatial@data@Dimnames

	Stdata@assays$SCT@counts=Stdata@misc$sct_counts
	Stdata@misc$sct_counts<-NULL
	if(!is.null(Stdata@misc$sct_genename)){
		print("Adding gene dimnames to SCT counts.")
		Stdata@assays$SCT@counts@Dimnames[[1]]  = Stdata@misc$sct_genename
	}
	if(!is.null(Stdata@misc$sct_cellname)){
		print("Adding cell dimnames to SCT counts.")
		Stdata@assays$SCT@counts@Dimnames[[2]]  = Stdata@misc$sct_cellname
	}
	#Stdata@assays$SCT@counts@Dimnames=Stdata@assays$Spatial@counts@Dimnames
	DefaultAssay(Stdata)<-"SCT"
	Stdata@misc$sct_genename<-NULL
	Stdata@misc$sct_cellname<-NULL
}
## add image
cell_coords=unique(Stdata@meta.data[, c('x', 'y')])
cell_coords['cell']=row.names(cell_coords)
cell_coords$x <- cell_coords$x - min(cell_coords$x) + 1
cell_coords$y <- cell_coords$y - min(cell_coords$y) + 1
# object of images$slice1@image, all illustrated as 1 since no concrete pic
tissue_lowres_image <- matrix(1, max(cell_coords$y), max(cell_coords$x))

# object of images$slice1@coordinates, concrete coordinate of X and Y
tissue_positions_list <- data.frame(row.names = cell_coords$cell,
                                    tissue = 1,
                                    row = cell_coords$y, col = cell_coords$x,
                                    imagerow = cell_coords$y, imagecol = cell_coords$x)
##@images$slice1@scale.factors
scalefactors_json <- toJSON(list(fiducial_diameter_fullres = 1,tissue_hires_scalef = 1,tissue_lowres_scalef = 1))

# generate object @images$slice1
generate_BGI_spatial <- function(image, scale.factors, tissue.positions, filter.matrix = TRUE){
  if (filter.matrix) {
    tissue.positions <- tissue.positions[which(tissue.positions$tissue == 1), , drop = FALSE]
  }
  unnormalized.radius <- scale.factors$fiducial_diameter_fullres * scale.factors$tissue_lowres_scalef
  spot.radius <- unnormalized.radius / max(dim(x = image))
  return(new(Class = 'VisiumV1',
             image = image,
             scale.factors = scalefactors(spot = scale.factors$tissue_hires_scalef,
                                          fiducial = scale.factors$fiducial_diameter_fullres,
                                          hires = scale.factors$tissue_hires_scalef,
                                          lowres = scale.factors$tissue_lowres_scalef),
             coordinates = tissue.positions,
             spot.radius = spot.radius))
}

BGI_spatial <- generate_BGI_spatial(image = tissue_lowres_image,
                                    scale.factors = fromJSON(scalefactors_json),
                                    tissue.positions = tissue_positions_list)
# can be thought of as a background of spatial
#' import image into seurat object
Stdata@images[['slice1']] <-BGI_spatial
Stdata@images$slice1@key<-"slice1_"
Stdata@images$slice1@assay<-"Spatial"
# conversion done, save
saveRDS(Stdata,outfile)
print("Finished RDS.")