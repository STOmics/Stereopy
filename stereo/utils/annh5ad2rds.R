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


Convert(infile, dest = "h5seurat",assay = "Spatial", overwrite = TRUE)
h5file<-paste(paste(unlist(strsplit(infile,"h5ad",fixed=TRUE)),collapse='h5ad'),"h5seurat",sep="")
Stdata <- LoadH5Seurat(h5file)
##增加SCT信息
#Stdata_SCT=CreateSCTAssayObject(
#  counts=NULL,
#  data=NULL,
#  scale.data = NULL,
#  umi.assay = "Spatial",
#  min.cells = 0,
#  min.features = 0,
#  SCTModel.list = NULL
#)

if(length(Stdata@misc$sct_data)>0){
	Stdata@assays$SCT=Stdata@assays$Spatial
	Stdata@assays$SCT@data=Stdata@misc$sct_data
	Stdata@assays$SCT@data@Dimnames=Stdata@assays$Spatial@data@Dimnames

	Stdata@assays$SCT@counts=Stdata@misc$sct_counts
	Stdata@assays$SCT@counts@Dimnames=Stdata@assays$Spatial@counts@Dimnames
}
##增加image
##
cell_coords=unique(Stdata@meta.data[, c('x', 'y')])
cell_coords['cell']=row.names(cell_coords)
cell_coords$x <- cell_coords$x - min(cell_coords$x) + 1
cell_coords$y <- cell_coords$y - min(cell_coords$y) + 1
##images$slice1@image对象，没有实际图片，全为1代替
tissue_lowres_image <- matrix(1, max(cell_coords$y), max(cell_coords$x))

##images$slice1@coordinates对象,实际X和Y坐标
tissue_positions_list <- data.frame(row.names = cell_coords$cell,
                                    tissue = 1,
                                    row = cell_coords$y, col = cell_coords$x,
                                    imagerow = cell_coords$y, imagecol = cell_coords$x)
##@images$slice1@scale.factors
scalefactors_json <- toJSON(list(fiducial_diameter_fullres = 1,tissue_hires_scalef = 1,tissue_lowres_scalef = 1))

##生成对象@images$slice1
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
#可以理解为构建一个spatial背景

#' import image into seurat object
Stdata@images[['slice1']] <-BGI_spatial
gc()
##转换完成，保持
saveRDS(Stdata,outfile)