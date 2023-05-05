library(anndata)
library(argparser)

args <- arg_parser("Converting .RDA(only one var typed as `SummarizedExperiment` file to .h5ad")
args <- add_argument(args, "--infile", help = "input .RDA file")
args <- add_argument(args, "--outfile", help = "output .h5ad file")
argv <- parse_args(args)

infile <- argv$infile
outfile <- argv$outfile

var.name <- load(file = infile, verbose = TRUE)
obj <- get(var.name)

if (class(obj)[1] != as.character("SummarizedExperiment")) {
  print("obj is not class `SummarizedExperiment`")
  quit("no", -1)
}

obj.ann.data <- AnnData(t(as.data.frame(obj@assays@data@listData[1])))
for (key in names(obj@colData@listData)) {
  obj.ann.data$obs[key] <- obj@colData@listData[key]
}

anndata::write_h5ad(anndata = obj.ann.data, filename = outfile, as_dense = "X")