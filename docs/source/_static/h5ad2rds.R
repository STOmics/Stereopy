library(dplyr)
library(rjson)
library(Seurat)
library(ggplot2)
library(argparser)
library(SeuratDisk)
library(hdf5r)
library(tools)
library(stringr)

args <- arg_parser("Converting h5ad file(.h5ad) to RDS.")
args <- add_argument(args, "--infile", help = "input .h5ad file")
# args <- add_argument(args, "--assay", help = "assay name")
args <- add_argument(args, "--outfile", help = "output RDS file")
argv <- parse_args(args)

infile=argv$infile
# assay=argv$assay
assay='Spatial'
outfile=argv$outfile

FileType <- function(file) {
  ext <- file_ext(x = file)
  ext <- ifelse(test = nchar(x = ext), yes = ext, no = basename(path = file))
  return(tolower(x = ext))
}

WriteMode <- function(overwrite = FALSE) {
  return(ifelse(test = overwrite, yes = 'w', no = 'w-'))
}

AttrExists <- function(x, name) {
  if (!inherits(x = x, what = c('H5File', 'H5Group', 'H5D'))) {
    stop("'x' must be an HDF5 file, group, or dataset", call. = FALSE)
  }
  exists <- x$attr_exists(attr_name = name)
  if (isTRUE(x = exists)) {
    space <- x$attr_open(attr_name = name)$get_space()
    exists <- !(length(x = space$dims) > 0 && space$dims == 0)
  }
  return(exists)
}

GuessDType <- function(x, stype = 'utf8', ...) {
  dtype <- guess_dtype(x = x, ...)
  if (inherits(x = dtype, what = 'H5T_STRING')) {
    dtype <- StringType(stype = stype)
  } else if (inherits(x = dtype, what = 'H5T_COMPOUND')) {
    cpd.dtypes <- dtype$get_cpd_types()
    for (i in seq_along(along.with = cpd.dtypes)) {
      if (inherits(x = cpd.dtypes[[i]], what = 'H5T_STRING')) {
        cpd.dtypes[[i]] <- StringType(stype = stype)
      }
    }
    dtype <- H5T_COMPOUND$new(
      labels = dtype$get_cpd_labels(),
      dtypes = cpd.dtypes,
      size = dtype$get_size()
    )
  } else if (inherits(x = dtype, what = 'H5T_LOGICAL')) {
    if (getOption(x = "SeuratDisk.dtypes.logical_to_int", default = TRUE)) {
      dtype <- guess_dtype(x = BoolToInt(x = x), ...)
    }
  }
  return(dtype)
}

StringType <- function(stype = c('utf8', 'ascii7')) {
  stype <- match.arg(arg = stype)
  return(switch(
    EXPR = stype,
    'utf8' = H5T_STRING$new(size = Inf)$set_cset(cset = h5const$H5T_CSET_UTF8),
    'ascii7' = H5T_STRING$new(size = 7L)
  ))
}

Exists <- function(x, name) {
  if (!inherits(x = x, what = c('H5File', 'H5Group'))) {
    stop("'x' must be an HDF5 file or group", call. = FALSE)
  }
  name <- unlist(x = strsplit(x = name[1], split = '/', fixed = TRUE))
  name <- Filter(f = nchar, x = name)
  path <- character(length = 1L)
  exists <- TRUE
  for (i in seq_along(along.with = name)) {
    path <- paste(path, name[i], sep = '/')
    if (!inherits(x = x, what = 'H5File')) {
      path <- gsub(pattern = '^/', replacement = '', x = path)
    }
    exists <- x$exists(name = path)
    if (isFALSE(x = exists)) {
      break
    }
  }
  return(exists)
}

IsDType <- function(x, dtype) {
  if (!inherits(x = x, what = 'H5D')) {
    stop("'IsDType' only works on HDF5 dataset", call. = FALSE)
  }
  dtypes <- unique(x = sapply(
    X = grep(pattern = '^H5T_', x = names(x = h5types), value = TRUE),
    FUN = function(i) {
      return(class(x = h5types[[i]])[1])
    },
    USE.NAMES = FALSE
  ))
  dtypes <- unique(x = c(dtypes, 'H5T_COMPOUND'))
  match.arg(arg = dtype, choices = dtypes, several.ok = TRUE)
  missing.dtypes <- setdiff(x = dtype, y = dtypes)
  if (length(x = missing.dtypes)) {
    dtype <- setdiff(x = dtype, y = missing.dtypes)
    if (!length(x = dtype)) {
      stop("None of the requested dtypes are valid HDF5 datatypes", call. = FALSE)
    } else {
      warning(
        "The following requested dtypes are not valid HDF5 datatypes: ",
        paste(missing.dtypes, sep = ", "),
        call. = FALSE,
        immediate. = TRUE
      )
    }
  }
  return(inherits(x = x$get_type(), what = dtype))
}

BoolToInt <- function(x) {
  x <- as.integer(x = x)
  x[which(x = is.na(x = x))] <- 2L
  return(x)
}

Convert <- function(source, dest, assay, overwrite = FALSE, verbose = TRUE, ...) {
  if (!missing(x = dest) && !is.character(x = dest)) {
    stop("'dest' must be a filename or type", call. = FALSE)
  }
  UseMethod(generic = 'Convert', object = source)
}

# Methods
Convert.character <- function(
  source,
  dest,
  assay,
  overwrite = FALSE,
  verbose = TRUE,
  ...
) {
  hfile <- Connect(filename = source, force = TRUE)
  if (missing(x = assay)) {
    assay <- tryCatch(
      expr = DefaultAssay(object = hfile),
      error = function(...) {
        warning(
          "'assay' not set, setting to 'RNA'",
          call. = FALSE,
          immediate. = TRUE
        )
        "RNA"
      }
    )
  }
  on.exit(expr = hfile$close_all())
  dfile <- Convert(
    source = hfile,
    dest = dest,
    assay = assay,
    overwrite = overwrite,
    verbose = verbose,
    ...
  )
  dfile$close_all()
  return(invisible(x = dfile$filename))
}

Convert.H5File <- function(
  source,
  dest = 'h5seurat',
  assay = 'RNA',
  overwrite = FALSE,
  verbose = TRUE,
  ...
) {
  stype <- FileType(file = source$filename)
  dtype <- FileType(file = dest)
  if (tolower(x = dest) == dtype) {
    dest <- paste(file_path_sans_ext(x = source$filename), dtype, sep = '.')
  }
  dfile <- switch(
    EXPR = stype,
    'h5ad' = switch(
      EXPR = dtype,
      'h5seurat' = H5ADToH5Seurat(
        source = source,
        dest = dest,
        assay = assay,
        overwrite = overwrite,
        verbose = verbose
      ),
      stop("Unable to convert H5AD files to ", dtype, " files", call. = FALSE)
    ),
    stop("Unknown file type: ", stype, call. = FALSE)
  )
  return(dfile)
}

Convert.h5Seurat <- function(
  source,
  dest = 'h5ad',
  assay = DefaultAssay(object = source),
  overwrite = FALSE,
  verbose = TRUE,
  ...
) {
  type <- FileType(file = dest)
  if (tolower(x = dest) == type) {
    dest <- paste(file_path_sans_ext(x = source$filename), type, sep = '.')
  }
  dfile <- switch(
    EXPR = type,
    'h5ad' = H5SeuratToH5AD(
      source = source,
      dest = dest,
      assay = assay,
      overwrite = overwrite,
      verbose = verbose
    ),
    stop("Unable to convert h5Seurat files to ", type, " files", call. = FALSE)
  )
  return(dfile)
}

# Implementations

H5ADToH5Seurat <- function(
  source,
  dest,
  assay = 'RNA',
  overwrite = FALSE,
  verbose = TRUE
) {
  if (file.exists(dest)) {
    if (overwrite) {
      file.remove(dest)
    } else {
      stop("Destination h5Seurat file exists", call. = FALSE)
    }
  }
  dfile <- h5Seurat$new(filename = dest, mode = WriteMode(overwrite = FALSE))
  # Get rownames from an H5AD data frame
  #
  # @param dset Name of data frame
  #
  # @return Returns the name of the dataset that contains the rownames
  #
  GetRownames <- function(dset) {
    if (inherits(x = source[[dset]], what = 'H5Group')) {
      # rownames <- if (source[[dset]]$attr_exists(attr_name = '_index')) {
      rownames <- if (isTRUE(x = AttrExists(x = source[[dset]], name = '_index'))) {
        h5attr(x = source[[dset]], which = '_index')
      } else if (source[[dset]]$exists(name = '_index')) {
        '_index'
      } else if (source[[dset]]$exists(name = 'index')) {
        'index'
      } else {
        stop("Cannot find rownames in ", dset, call. = FALSE)
      }
    } else {
      # TODO: fix this
      stop("Don't know how to handle datasets", call. = FALSE)
      # rownames(x = source[[dset]])
    }
    return(rownames)
  }
  ColToFactor <- function(dfgroup) {
    if (dfgroup$exists(name = '__categories')) {
      for (i in names(x = dfgroup[['__categories']])) {
        tname <- basename(path = tempfile(tmpdir = ''))
        dfgroup$obj_copy_to(dst_loc = dfgroup, dst_name = tname, src_name = i)
        dfgroup$link_delete(name = i)
        # Because AnnData stores logicals as factors, but have too many levels
        # for factors
        bool.check <- dfgroup[['__categories']][[i]]$dims == 2
        if (isTRUE(x = bool.check)) {
          bool.check <- all(sort(x = dfgroup[['__categories']][[i]][]) == c('False', 'True'))
        }
        if (isTRUE(x = bool.check)) {
          dfgroup$create_dataset(
            name = i,
            robj = dfgroup[[tname]][] + 1L,
            dtype = dfgroup[[tname]]$get_type()
          )
        } else {
          dfgroup$create_group(name = i)
          dfgroup[[i]]$create_dataset(
            name = 'values',
            robj = dfgroup[[tname]][] + 1L,
            dtype = dfgroup[[tname]]$get_type()
          )
          if (IsDType(x = dfgroup[['__categories']][[i]], dtype = 'H5T_STRING')) {
            dfgroup$obj_copy_to(
              dst_loc = dfgroup,
              dst_name = paste0(i, '/levels'),
              src_name = paste0('__categories/', i)
            )
          } else {
            dfgroup[[i]]$create_dataset(
              name = 'levels',
              robj = as.character(x = dfgroup[[H5Path('__categories', i)]][]),
              dtype = StringType()
            )
          }
        }
        dfgroup$link_delete(name = tname)
      }
      dfgroup$link_delete(name = '__categories')
    }
    return(invisible(x = NULL))
  }
  ds.map <- c(
    scale.data = if (inherits(x = source[['X']], what = 'H5D')) {
      'X'
    } else {
      NULL
    },
    data = if (inherits(x = source[['X']], what = 'H5D') && source$exists(name = 'raw')) {
      'raw/X'
    } else {
      'X'
    },
    counts = if (source$exists(name = 'raw')) {
      'raw/X'
    } else {
      'X'
    }
  )
  # Add assay data
  assay.group <- dfile[['assays']]$create_group(name = assay)
  for (i in seq_along(along.with = ds.map)) {
    if (verbose) {
      message("Adding ", ds.map[[i]], " as ", names(x = ds.map)[i])
    }
    dst <- names(x = ds.map)[i]
    assay.group$obj_copy_from(
      src_loc = source,
      src_name = ds.map[[i]],
      dst_name = dst
    )
    # if (assay.group[[dst]]$attr_exists(attr_name = 'shape')) {
    if (isTRUE(x = AttrExists(x = assay.group[[dst]], name = 'shape'))) {
      dims <- rev(x = h5attr(x = assay.group[[dst]], which = 'shape'))
      assay.group[[dst]]$create_attr(
        attr_name = 'dims',
        robj = dims,
        dtype = GuessDType(x = dims)
      )
      assay.group[[dst]]$attr_delete(attr_name = 'shape')
    }
  }
  features.source <- ifelse(
    test = source$exists(name = 'raw') && source$exists(name = 'raw/var'),
    yes = 'raw/var',
    no = 'var'
  )
  if (inherits(x = source[[features.source]], what = 'H5Group')) {
    features.dset <- GetRownames(dset = features.source)
    assay.group$obj_copy_from(
      src_loc = source,
      src_name = paste(features.source, features.dset, sep = '/'),
      dst_name = 'features'
    )
  } else {
    tryCatch(
      expr = assay.group$create_dataset(
        name = 'features',
        robj = rownames(x = source[[features.source]]),
        dtype = GuessDType(x = "")
      ),
      error = function(...) {
        stop("Cannot find feature names in this H5AD file", call. = FALSE)
      }
    )
  }
  scaled <- !is.null(x = ds.map['scale.data']) && !is.na(x = ds.map['scale.data'])
  if (scaled) {
    if (inherits(x = source[['var']], what = 'H5Group')) {
      scaled.dset <- GetRownames(dset = 'var')
      assay.group$obj_copy_from(
        src_loc = source,
        src_name = paste0('var/', scaled.dset),
        dst_name = 'scaled.features'
      )
    } else {
      tryCatch(
        expr = assay.group$create_dataset(
          name = 'scaled.features',
          robj = rownames(x = source[['var']]),
          dtype = GuessDType(x = "")
        ),
        error = function(...) {
          stop("Cannot find scaled features in this H5AD file", call. = FALSE)
        }
      )
    }
  }
  assay.group$create_attr(
    attr_name = 'key',
    robj = paste0(tolower(x = assay), '_'),
    dtype = GuessDType(x = assay)
  )
  # Set default assay
  DefaultAssay(object = dfile) <- assay
  # Add feature-level metadata
  if (!getOption(x = "SeuratDisk.dtypes.dataframe_as_group", default = FALSE)) {
    warning(
      "Adding feature-level metadata as a compound is not yet supported",
      call. = FALSE,
      immediate. = TRUE
    )
  }
  # TODO: Support compound metafeatures
  if (Exists(x = source, name = 'raw/var')) {
    if (inherits(x = source[['raw/var']], what = 'H5Group')) {
      if (verbose) {
        message("Adding meta.features from raw/var")
      }
      assay.group$obj_copy_from(
        src_loc = source,
        src_name = 'raw/var',
        dst_name = 'meta.features'
      )
      if (scaled) {
        features.use <- assay.group[['features']][] %in% assay.group[['scaled.features']][]
        features.use <- which(x = features.use)
        meta.scaled <- names(x = source[['var']])
        meta.scaled <- meta.scaled[!meta.scaled %in% c('__categories', scaled.dset)]
        for (mf in meta.scaled) {
          if (!mf %in% names(x = assay.group[['meta.features']])) {
            if (verbose) {
              message("Adding ", mf, " from scaled feature-level metadata")
            }
            assay.group[['meta.features']]$create_dataset(
              name = mf,
              dtype = source[['var']][[mf]]$get_type(),
              space = H5S$new(dims = assay.group[['features']]$dims)
            )
          } else if (verbose) {
            message("Merging ", mf, " from scaled feature-level metadata")
          }
          assay.group[['meta.features']][[mf]][features.use] <- source[['var']][[mf]]$read()
        }
      }
    } else {
      warning(
        "Cannot yet add feature-level metadata from compound datasets",
        call. = FALSE,
        immediate. = TRUE
      )
      assay.group$create_group(name = 'meta.features')
    }
  } else {
    if (inherits(x = source[['var']], what = 'H5Group')) {
      if (verbose) {
        message("Adding meta.features from var")
      }
      assay.group$obj_copy_from(
        src_loc = source,
        src_name = 'var',
        dst_name = 'meta.features'
      )
    } else {
      warning(
        "Cannot yet add feature-level metadata from compound datasets",
        call. = FALSE,
        immediate. = TRUE
      )
      assay.group$create_group(name = 'meta.features')
    }
  }
  ColToFactor(dfgroup = assay.group[['meta.features']])
  # if (assay.group[['meta.features']]$attr_exists(attr_name = 'column-order')) {
  if (isTRUE(x = AttrExists(x = assay.group[['meta.features']], name = 'column-order'))) {
    colnames <- h5attr(
      x = assay.group[['meta.features']],
      which = 'column-order'
    )
    assay.group[['meta.features']]$create_attr(
      attr_name = 'colnames',
      robj = colnames,
      dtype = GuessDType(x = colnames)
    )
  }
  if (inherits(x = source[['var']], what = 'H5Group')) {
    assay.group[['meta.features']]$link_delete(name = GetRownames(dset = 'var'))
  }
  # Add cell-level metadata
  if (source$exists(name = 'obs') && inherits(x = source[['obs']], what = 'H5Group')) {
    if (!source[['obs']]$exists(name = '__categories') && !getOption(x = "SeuratDisk.dtypes.dataframe_as_group", default = TRUE)) {
      warning(
        "Conversion from H5AD to h5Seurat allowing compound datasets is not yet implemented",
        call. = FALSE,
        immediate. = TRUE
      )
    }
    meta=dfile$create_group(name='meta.data')
    for (metainfo in names(x = source[['obs']])) {
      meta$obj_copy_from(
        src_loc = source,
        src_name = paste0('obs/',metainfo),
        dst_name = metainfo
      )
    }
    ColToFactor(dfgroup = meta)
    # if (dfile[['meta.data']]$attr_exists(attr_name = 'column-order')) {
    if (isTRUE(x = AttrExists(x = meta, name = 'column-order'))) {
      colnames <- h5attr(x = meta, which = 'column-order')
      meta$create_attr(
        attr_name = 'colnames',
        robj = colnames,
        dtype = GuessDType(x = colnames)
      )
    }
    rownames <- GetRownames(dset = 'obs')
    dfile$obj_copy_from(
      src_loc = dfile,
      src_name = paste0('meta.data/', rownames),
      dst_name = 'cell.names'
    )
  } else {
    warning(
      "No cell-level metadata present, creating fake cell names",
      call. = FALSE,
      immediate. = TRUE
    )
    ncells <- if (inherits(x = assay.group[['data']], what = 'H5Group')) {
      assay.group[['data/indptr']]$dims - 1
    } else {
      assay.group[['data']]$dims[2]
    }
    dfile$create_group(name = 'meta.data')
    dfile$create_dataset(
      name = 'cell.names',
      robj = paste0('Cell', seq.default(from = 1, to = ncells)),
      dtype = GuessDType(x = 'Cell1')
    )
  }
  # Add dimensional reduction information
  if (source$exists(name = 'obsm')) {
    # Add cell embeddings
    if (inherits(x = source[['obsm']], what = 'H5Group')) {
      for (reduc in names(x = source[['obsm']])) {
        if (reduc == 'cell_border') {
            next
        }
        sreduc <- gsub(pattern = '^X_', replacement = '', x = reduc)
        reduc.group <- dfile[['reductions']]$create_group(name = sreduc)
        message("Adding ", reduc, " as cell embeddings for ", sreduc)
        Transpose(
          x = source[['obsm']][[reduc]],
          dest = reduc.group,
          dname = 'cell.embeddings',
          verbose = FALSE
        )
        reduc.group$create_group(name = 'misc')
        reduc.group$create_attr(
          attr_name = 'active.assay',
          robj = assay,
          dtype = GuessDType(x = assay)
        )
        key <- paste0(
          if (grepl(pattern = 'pca', x = sreduc, ignore.case = TRUE)) {
            'PC'
          } else if (grepl(pattern = 'tsne', x = sreduc, ignore.case = TRUE)) {
            'tSNE'
          } else {
            sreduc
          },
          '_'
        )
        reduc.group$create_attr(
          attr_name = 'key',
          robj = key,
          dtype = GuessDType(x = reduc)
        )
        global <- BoolToInt(x = grepl(
          pattern = 'tsne|umap',
          x = sreduc,
          ignore.case = TRUE
        ))
        reduc.group$create_attr(
          attr_name = 'global',
          robj = global,
          dtype = GuessDType(x = global)
        )
      }
    } else {
      warning(
        "Reading compound dimensional reductions not yet supported, please update your H5AD file",
        call. = FALSE,
        immediate. = TRUE
      )
    }
    # Add feature loadings
    if (source$exists(name = 'varm')) {
      if (inherits(x = source[['varm']], what = 'H5Group')) {
        for (reduc in names(x = source[['varm']])) {
          sreduc <- switch(EXPR = reduc, 'PCs' = 'pca', tolower(x = reduc))
          if (!isTRUE(x = sreduc %in% names(x = dfile[['reductions']]))) {
            warning(
              "Cannot find a reduction named ",
              sreduc,
              " (",
              reduc,
              " in varm)",
              call. = FALSE,
              immediate. = TRUE
            )
            next
          }
          if (isTRUE(x = verbose)) {
            message("Adding ", reduc, " as feature loadings fpr ", sreduc)
          }
          Transpose(
            x = source[['varm']][[reduc]],
            dest = dfile[['reductions']][[sreduc]],
            dname = 'feature.loadings',
            verbose = FALSE
          )
          reduc.features <- dfile[['reductions']][[sreduc]][['feature.loadings']]$dims[1]
          assay.features <- if (assay.group[['features']]$dims == reduc.features) {
            'features'
          } else if (assay.group$exists(name = 'scaled.features') && assay.group[['scaled.features']]$dims == reduc.features) {
            'scaled.features'
          } else {
            NULL
          }
          if (is.null(x = assay.features)) {
            warning(
              "Cannot find features for feature loadings, will not be able to load",
              call. = FALSE,
              immediate. = TRUE
            )
          } else {
            dfile[['reductions']][[sreduc]]$obj_copy_from(
              src_loc = assay.group,
              src_name = assay.features,
              dst_name = 'features'
            )
          }
        }
      } else {
        warning(
          "Reading compound dimensional reductions not yet supported",
          call. = FALSE,
          immediate. = TRUE
        )
      }
    }
    # Add miscellaneous information
    if (source$exists(name = 'uns')) {
      for (reduc in names(x = source[['uns']])) {
        if (!isTRUE(x = reduc %in% names(x = dfile[['reductions']]))) {
          next
        }
        if (verbose) {
          message("Adding miscellaneous information for ", reduc)
        }
        dfile[['reductions']][[reduc]]$link_delete(name = 'misc')
        dfile[['reductions']][[reduc]]$obj_copy_from(
          src_loc = source[['uns']],
          src_name = reduc,
          dst_name = 'misc'
        )
        if ('variance' %in% names(x = dfile[['reductions']][[reduc]][['misc']])) {
          if (verbose) {
            message("Adding standard deviations for ", reduc)
          }
          dfile[['reductions']][[reduc]]$create_dataset(
            name = 'stdev',
            robj = sqrt(x = dfile[['reductions']][[reduc]][['misc']][['variance']][]),
            dtype = GuessDType(x = 1.0)
          )
        }
      }
    }
  }
  # Add project and cell identities
  Project(object = dfile) <- 'AnnData'
  idents <- dfile$create_group(name = 'active.ident')
  idents$create_dataset(
    name = 'values',
    dtype = GuessDType(x = 1L),
    space = H5S$new(dims = dfile[['cell.names']]$dims)
  )
  idents$create_dataset(
    name = 'levels',
    robj = 'AnnData',
    dtype = GuessDType(x = 'AnnData')
  )
  idents[['values']]$write(
    args = list(seq.default(from = 1, to = idents[['values']]$dims)),
    value = 1L
  )
  # Add nearest-neighbor graph
  if (Exists(x = source, name = 'uns/neighbors/distances')) {
    graph.name <- paste(
      assay,
      ifelse(
        test = source$exists(name = 'uns/neighbors/params/method'),
        yes = source[['uns/neighbors/params/method']][1],
        no = 'anndata'
      ),
      sep = '_'
    )
    if (verbose) {
      message("Saving nearest-neighbor graph as ", graph.name)
    }
    dfile[['graphs']]$obj_copy_from(
      src_loc = source,
      src_name = 'uns/neighbors/distances',
      dst_name = graph.name
    )
    # if (dfile[['graphs']][[graph.name]]$attr_exists(attr_name = 'shape')) {
    if (isTRUE(x = AttrExists(x = dfile[['graphs']], name = 'shape'))) {
      dfile[['graphs']][[graph.name]]$create_attr(
        attr_name = 'dims',
        robj = h5attr(x = dfile[['graphs']][[graph.name]], which = 'shape'),
        dtype = GuessDType(x = h5attr(
          x = dfile[['graphs']][[graph.name]],
          which = 'shape'
        ))
      )
      dfile[['graphs']][[graph.name]]$attr_delete(attr_name = 'shape')
    }
    dfile[['graphs']][[graph.name]]$create_attr(
      attr_name = 'assay.used',
      robj = assay,
      dtype = GuessDType(x = assay)
    )
  }
  # Add miscellaneous information
  if (source$exists(name = 'uns')) {
    misc <- setdiff(
      x = names(x = source[['uns']]),
      y = c('neighbors', names(x = dfile[['reductions']]))
    )
    for (i in misc) {
      if (verbose) {
        message("Adding ", i, " to miscellaneous data")
      }
      dfile[['misc']]$obj_copy_from(
        src_loc = source[['uns']],
        src_name = i,
        dst_name = i
      )
    }
  }
  # Add layers
  if (Exists(x = source, name = 'layers')) {
    slots <- c('data')
    if (!isTRUE(x = scaled)) {
      slots <- c(slots, 'counts')
    }
    for (layer in names(x = source[['layers']])) {
      layer.assay <- dfile[['assays']]$create_group(name = layer)
      layer.assay$obj_copy_from(
        src_loc = dfile[['assays']][[assay]],
        src_name = 'features',
        dst_name = 'features'
      )
      layer.assay$create_attr(
        attr_name = 'key',
        robj = UpdateKey(key = layer),
        dtype = GuessDType(x = layer)
      )
      for (slot in slots) {
        if (verbose) {
          message("Adding layer ", layer, " as ", slot, " in assay ", layer)
        }
        layer.assay$obj_copy_from(
          src_loc = source[['layers']],
          src_name = layer,
          dst_name = slot
        )
        # if (layer.assay[[slot]]$attr_exists(attr_name = 'shape')) {
        if (isTRUE(x = AttrExists(x = layer.assay[[slot]], name = 'shape'))) {
          dims <- rev(x = h5attr(x = layer.assay[[slot]], which = 'shape'))
          layer.assay[[slot]]$create_attr(
            attr_name = 'dims',
            robj = dims,
            dtype = GuessDType(x = dims)
          )
          layer.assay[[slot]]$attr_delete(attr_name = 'shape')
        }
      }
    }
  }
  return(dfile)
}

#' Convert h5Seurat files to H5AD files
#'
#' @inheritParams Convert
#'
#' @return Returns a handle to \code{dest} as an \code{\link[hdf5r]{H5File}}
#' object
#'
#' @section h5Seurat to AnnData/H5AD:
#' The h5Seurat to AnnData/H5AD conversion will try to automatically fill in
#' datasets based on data presence. Data presense is determined by the h5Seurat
#' index (\code{source$index()}). It works in the following manner:
#' \subsection{Assay data}{
#'  \itemize{
#'   \item \code{X} will be filled with \code{scale.data} if \code{scale.data}
#'   is present; otherwise, it will be filled with \code{data}
#'   \item \code{var} will be filled with \code{meta.features} \strong{only} for
#'   the features present in \code{X}; for example, if \code{X} is filled with
#'   \code{scale.data}, then \code{var} will contain only features that have
#'   been scaled
#'   \item \code{raw.X} will be filled with \code{data} if \code{X} is filled
#'   with \code{scale.data}; otherwise, it will be filled with \code{counts}. If
#'   \code{counts} is not present, then \code{raw} will not be filled
#'   \item \code{raw.var} will be filled with \code{meta.features} with the
#'   features present in \code{raw.X}; if \code{raw.X} is not filled, then
#'   \code{raw.var} will not be filled
#'  }
#' }
#' \subsection{Cell-level metadata}{
#'  Cell-level metadata is added to \code{obs}
#' }
#' \subsection{Dimensional reduction information}{
#'  Only dimensional reductions associated with \code{assay} or marked as
#'  \link[Seurat:IsGlobal]{global} will be transfered to the H5AD file. For
#'  every reduction \code{reduc}:
#'  \itemize{
#'   \item cell embeddings are placed in \code{obsm} and renamed to
#'   \code{X_reduc}
#'   \item feature loadings, if present, are placed in \code{varm} and renamed
#'   to either \dQuote{PCs} if \code{reduc} is \dQuote{pca} otherwise
#'   \code{reduc} in all caps
#'  }
#'  For example, if \code{reduc} is \dQuote{ica}, then cell embeddings will be
#'  \dQuote{X_ica} in \code{obsm} and feature loaodings, if present, will be
#'  \dQuote{ICA} in \code{varm}
#' }
#' \subsection{Nearest-neighbor graphs}{
#'  If a nearest-neighbor graph is associated with \code{assay}, it will be
#'  added to \code{uns/neighbors/distances}; if more than one graph is present,
#'  then \strong{only} the last graph according to the index will be added.
#' }
#' \subsection{Layers}{
#'  Data from other assays can be added to \code{layers} if they have the same
#'  shape as \code{X} (same number of cells and features). To determine this,
#'  the shape of each alternate assays's \code{scale.data} and \code{data} slots
#'  are determined. If they are the same shape as \code{X}, then that slot
#'  (\code{scale.data} is given priority over \code{data}) will be added as a
#'  layer named the name of the assay (eg. \dQuote{SCT}). In addition, the
#'  features names will be added to \code{var} as \code{assay_features}
#'  (eg. \dQuote{SCT_features}).
#' }
#'
#' @keywords internal
#'
H5SeuratToH5AD <- function(
  source,
  dest,
  assay = DefaultAssay(object = source),
  overwrite = FALSE,
  verbose = TRUE
) {
  if (file.exists(dest)) {
    if (overwrite) {
      file.remove(dest)
    } else {
      stop("Destination H5AD file exists", call. = FALSE)
    }
  }
  rownames <- '_index'
  dfile <- H5File$new(filename = dest, mode = WriteMode(overwrite = FALSE))
  # Transfer data frames from h5Seurat files to H5AD files
  #
  # @param src Source dataset
  # @param dname Name of destination
  # @param index Integer values of rows to take
  #
  # @return Invisibly returns \code{NULL}
  #
  TransferDF <- function(src, dname, index) {
    if (verbose) {
      message("Transfering ", basename(path = src$get_obj_name()), " to ", dname)
    }
    if (inherits(x = src, what = 'H5D')) {
      CompoundToGroup(
        src = src,
        dest = dfile,
        dst.name = dname,
        order = 'column-order',
        index = index
      )
    } else {
      dfile$create_group(name = dname)
      for (i in src$names) {
        if (IsFactor(x = src[[i]])) {
          dfile[[dname]]$create_dataset(
            name = i,
            robj = src[[i]][['values']][index] - 1L,
            dtype = src[[H5Path(i, 'values')]]$get_type()
          )
          if (!dfile[[dname]]$exists(name = '__categories')) {
            dfile[[dname]]$create_group(name = '__categories')
          }
          dfile[[dname]][['__categories']]$create_dataset(
            name = i,
            robj = src[[i]][['levels']][],
            dtype = src[[H5Path(i, 'levels')]]$get_type()
          )
        } else {
          dfile[[dname]]$create_dataset(
            name = i,
            robj = src[[i]][index],
            dtype = src[[i]]$get_type()
          )
        }
      }
      if (src$attr_exists(attr_name = 'colnames')) {
        dfile[[dname]]$create_attr(
          attr_name = 'column-order',
          robj = h5attr(x = src, which = 'colnames'),
          dtype = GuessDType(x = h5attr(x = src, which = 'colnames'))
        )
      }
      encoding.info <- c('type' = 'dataframe', 'version' = '0.1.0')
      names(x = encoding.info) <- paste0('encoding-', names(x = encoding.info))
      for (i in seq_along(along.with = encoding.info)) {
        attr.name <- names(x = encoding.info)[i]
        attr.value <- encoding.info[i]
        if (dfile[[dname]]$attr_exists(attr_name = attr.name)) {
          dfile[[dname]]$attr_delete(attr_name = attr.name)
        }
        dfile[[dname]]$create_attr(
          attr_name = attr.name,
          robj = attr.value,
          dtype = GuessDType(x = attr.value),
          space = Scalar()
        )
      }
    }
    return(invisible(x = NULL))
  }
  # Because AnnData can't figure out that sparse matrices are stored as groups
  AddEncoding <- function(dname) {
    encoding.info <- c('type' = 'csr_matrix', 'version' = '0.1.0')
    names(x = encoding.info) <- paste0('encoding-', names(x = encoding.info))
    if (inherits(x = dfile[[dname]], what = 'H5Group')) {
      for (i in seq_along(along.with = encoding.info)) {
        attr.name <- names(x = encoding.info)[i]
        attr.value <- encoding.info[i]
        if (dfile[[dname]]$attr_exists(attr_name = attr.name)) {
          dfile[[dname]]$attr_delete(attr_name = attr.name)
        }
        dfile[[dname]]$create_attr(
          attr_name = attr.name,
          robj = attr.value,
          dtype = GuessDType(x = attr.value),
          space = Scalar()
        )
      }
      # dfile[[dname]]$create_attr(
      #   attr_name = 'encoding-type',
      #   robj = 'csr_matrix',
      #   dtype = StringType(),
      #   space = Scalar()
      # )
      # dfile[[dname]]$create_attr(
      #   attr_name = 'encoding-version',
      #   robj = '0.1.0',
      #   dtype = StringType(),
      #   space = Scalar()
      # )
    }
    return(invisible(x = NULL))
  }
  # Add assay data
  assay.group <- source[['assays']][[assay]]
  if (source$index()[[assay]]$slots[['scale.data']]) {
    x.data <- 'scale.data'
    raw.data <- 'data'
  } else {
    x.data <- 'data'
    raw.data <- if (source$index()[[assay]]$slots[['counts']]) {
      'counts'
    } else {
      NULL
    }
  }
  if (verbose) {
    message("Adding ", x.data, " from ", assay, " as X")
  }
  assay.group$obj_copy_to(dst_loc = dfile, dst_name = 'X', src_name = x.data)
  if (dfile[['X']]$attr_exists(attr_name = 'dims')) {
    dims <- h5attr(x = dfile[['X']], which = 'dims')
    dfile[['X']]$create_attr(
      attr_name = 'shape',
      robj = rev(x = dims),
      dtype = GuessDType(x = dims)
    )
    dfile[['X']]$attr_delete(attr_name = 'dims')
  }
  AddEncoding(dname = 'X')
  x.features <- switch(
    EXPR = x.data,
    'scale.data' = which(x = assay.group[['features']][] %in% assay.group[['scaled.features']][]),
    seq.default(from = 1, to = assay.group[['features']]$dims)
  )
  # Add meta.features
  if (assay.group$exists(name = 'meta.features')) {
    TransferDF(
      src = assay.group[['meta.features']],
      dname = 'var',
      index = x.features
    )
  } else {
    dfile$create_group(name = 'var')
  }
  # Add feature names
  if (Exists(x = dfile[['var']], name = rownames)) {
    dfile[['var']]$link_delete(name = rownames)
  }
  dfile[['var']]$create_dataset(
    name = rownames,
    robj = assay.group[['features']][x.features],
    dtype = GuessDType(x = assay.group[['features']][1])
  )
  dfile[['var']]$create_attr(
    attr_name = rownames,
    robj = rownames,
    dtype = GuessDType(x = rownames),
    space = Scalar()
  )
  # Because AnnData requries meta.features and can't build an empty data frame
  if (!dfile[['var']]$attr_exists(attr_name = 'column-order')) {
    var.cols <- setdiff(
      x = names(x = dfile[['var']]),
      y = c(rownames, '__categories')
    )
    if (!length(x = var.cols)) {
      var.cols <- 'features'
      dfile[['var']]$obj_copy_to(
        dst_loc = dfile[['var']],
        dst_name = var.cols,
        src_name = rownames
      )
    }
    dfile[['var']]$create_attr(
      attr_name = 'column-order',
      robj = var.cols,
      dtype = GuessDType(x = var.cols)
    )
  }
  
  # Add encoding, to ensure compatibility with python's anndata > 0.8.0:
  encoding.info <- c('type' = 'dataframe', 'version' = '0.1.0')
  names(x = encoding.info) <- paste0('encoding-', names(x = encoding.info))
  for (i in seq_along(along.with = encoding.info)) {
    attr.name <- names(x = encoding.info)[i]
    attr.value <- encoding.info[i]
    if (dfile[['var']]$attr_exists(attr_name = attr.name)) {
      dfile[['var']]$attr_delete(attr_name = attr.name)
    }
    dfile[['var']]$create_attr(
      attr_name = attr.name,
      robj = attr.value,
      dtype = GuessDType(x = attr.value),
      space = Scalar()
    )
  }
  
  # Add raw
  if (!is.null(x = raw.data)) {
    if (verbose) {
      message("Adding ", raw.data, " from ", assay, " as raw")
    }
    dfile$create_group(name = 'raw')
    assay.group$obj_copy_to(
      dst_loc = dfile[['raw']],
      dst_name = 'X',
      src_name = raw.data
    )
    if (dfile[['raw/X']]$attr_exists(attr_name = 'dims')) {
      dims <- h5attr(x = dfile[['raw/X']], which = 'dims')
      dfile[['raw/X']]$create_attr(
        attr_name = 'shape',
        robj = rev(x = dims),
        dtype = GuessDType(x = dims)
      )
      dfile[['raw/X']]$attr_delete(attr_name = 'dims')
    }
    AddEncoding(dname = 'raw/X')
    # Add meta.features
    if (assay.group$exists(name = 'meta.features')) {
      TransferDF(
        src = assay.group[['meta.features']],
        dname = 'raw/var',
        index = seq.default(from = 1, to = assay.group[['features']]$dims)
      )
    } else {
      dfile[['raw']]$create_group(name = 'var')
    }
    # Add feature names
    if (Exists(x = dfile[['raw/var']], name = rownames)) {
      dfile[['raw/var']]$link_delete(name = rownames)
    }
    dfile[['raw/var']]$create_dataset(
      name = rownames,
      robj = assay.group[['features']][],
      dtype = GuessDType(x = assay.group[['features']][1])
    )
    dfile[['raw/var']]$create_attr(
      attr_name = rownames,
      robj = rownames,
      dtype = GuessDType(x = rownames),
      space = Scalar()
    )
  }
  # Add cell-level metadata
  TransferDF(
    src = source[['meta.data']],
    dname = 'obs',
    index = seq.default(from = 1, to = length(x = Cells(x = source)))
  )
  if (Exists(x = dfile[['obs']], name = rownames)) {
    dfile[['obs']]$link_delete(name = rownames)
  }
  dfile[['obs']]$create_dataset(
    name = rownames,
    robj = Cells(x = source),
    dtype = GuessDType(x = Cells(x = source))
  )
  dfile[['obs']]$create_attr(
    attr_name = rownames,
    robj = rownames,
    dtype = GuessDType(x = rownames),
    space = Scalar()
  )
  # Add dimensional reduction information
  obsm <- dfile$create_group(name = 'obsm')
  varm <- dfile$create_group(name = 'varm')
  reductions <- source$index()[[assay]]$reductions
  for (reduc in names(x = reductions)) {
    if (verbose) {
      message("Adding dimensional reduction information for ", reduc)
    }
    Transpose(
      x = source[[H5Path('reductions', reduc, 'cell.embeddings')]],
      dest = obsm,
      dname = paste0('X_', reduc),
      verbose = FALSE
    )
    if (reductions[[reduc]]['feature.loadings']) {
      if (verbose) {
        message("Adding feature loadings for ", reduc)
      }
      loadings <- source[['reductions']][[reduc]][['feature.loadings']]
      reduc.features <- loadings$dims[1]
      x.features <- dfile[['var']][[rownames]]$dims
      varm.name <- switch(EXPR = reduc, 'pca' = 'PCs', toupper(x = reduc))
      # Because apparently AnnData requires nPCs == nrow(X)
      if (reduc.features < x.features) {
        pad <- paste0('pad_', varm.name)
        PadMatrix(
          src = loadings,
          dest = dfile[['varm']],
          dname = pad,
          dims = c(x.features, loadings$dims[2]),
          index = list(
            match(
              x = source[['reductions']][[reduc]][['features']][],
              table = dfile[['var']][[rownames]][]
            ),
            seq.default(from = 1, to = loadings$dims[2])
          )
        )
        loadings <- dfile[['varm']][[pad]]
      }
      Transpose(x = loadings, dest = varm, dname = varm.name, verbose = FALSE)
      if (reduc.features < x.features) {
        dfile$link_delete(name = loadings$get_obj_name())
      }
    }
  }
  # Add global dimensional reduction information
  global.reduc <- source$index()[['global']][['reductions']]
  for (reduc in global.reduc) {
    if (reduc %in% names(x = reductions)) {
      next
    } else if (verbose) {
      message("Adding dimensional reduction information for ", reduc, " (global)")
    }
    Transpose(
      x = source[[H5Path('reductions', reduc, 'cell.embeddings')]],
      dest = obsm,
      dname = paste0('X_', reduc),
      verbose = FALSE
    )
  }
  # Create uns
  dfile$create_group(name = 'uns')
  # Add graph
  graph <- source$index()[[assay]]$graphs
  graph <- graph[length(x = graph)]
  if (!is.null(x = graph)) {
    if (verbose) {
      message("Adding ", graph, " as neighbors")
    }
    dgraph <- dfile[['uns']]$create_group(name = 'neighbors')
    source[['graphs']]$obj_copy_to(
      dst_loc = dgraph,
      dst_name = 'distances',
      src_name = graph
    )
    if (source[['graphs']][[graph]]$attr_exists(attr_name = 'dims')) {
      dims <- h5attr(x = source[['graphs']][[graph]], which = 'dims')
      dgraph[['distances']]$create_attr(
        attr_name = 'shape',
        robj = rev(x = dims),
        dtype = GuessDType(x = dims)
      )
    }
    AddEncoding(dname = 'uns/neighbors/distances')
    # Add parameters
    dgraph$create_group(name = 'params')
    dgraph[['params']]$create_dataset(
      name = 'method',
      robj = gsub(pattern = paste0('^', assay, '_'), replacement = '', x = graph),
      dtype = GuessDType(x = graph)
    )
    cmdlog <- paste(
      paste0('FindNeighbors.', assay),
      unique(x = c(names(x = reductions), source$index()$global$reductions)),
      sep = '.',
      collapse = '|'
    )
    cmdlog <- grep(
      pattern = cmdlog,
      x = names(x = source[['commands']]),
      value = TRUE
    )
    if (length(x = cmdlog) > 1) {
      timestamps <- sapply(
        X = cmdlog,
        FUN = function(cmd) {
          ts <- if (source[['commands']][[cmd]]$attr_exists(attr_name = 'time.stamp')) {
            h5attr(x = source[['commands']][[cmd]], which = 'time.stamp')
          } else {
            NULL
          }
          return(ts)
        },
        simplify = TRUE,
        USE.NAMES = FALSE
      )
      timestamps <- Filter(f = Negate(f = is.null), x = timestamps)
      cmdlog <- cmdlog[order(timestamps, decreasing = TRUE)][1]
    }
    if (length(x = cmdlog) && !is.na(x = cmdlog)) {
      cmdlog <- source[['commands']][[cmdlog]]
      if ('k.param' %in% names(x = cmdlog)) {
        dgraph[['params']]$obj_copy_from(
          src_loc = cmdlog,
          src_name = 'k.param',
          dst_name = 'n_neighbors'
        )
      }
    }
  }
  # Add layers
  other.assays <- setdiff(
    x = names(x = source$index()),
    y = c(assay, 'global', 'no.assay')
  )
  if (length(x = other.assays)) {
    x.dims <- Dims(x = dfile[['X']])
    layers <- dfile$create_group(name = 'layers')
    for (other in other.assays) {
      layer.slot <- NULL
      for (slot in c('scale.data', 'data')) {
        slot.dims <- if (source$index()[[other]]$slots[[slot]]) {
          Dims(x = source[['assays']][[other]][[slot]])
        } else {
          NA_integer_
        }
        if (isTRUE(all.equal(slot.dims, x.dims))) {
          layer.slot <- slot
          break
        }
      }
      if (!is.null(x = layer.slot)) {
        if (verbose) {
          message("Adding ", layer.slot, " from ", other, " as a layer")
        }
        layers$obj_copy_from(
          src_loc = source[['assays']][[other]],
          src_name = layer.slot,
          dst_name = other
        )
        if (layers[[other]]$attr_exists(attr_name = 'dims')) {
          dims <- h5attr(x = layers[[other]], which = 'dims')
          layers[[other]]$create_attr(
            attr_name = 'shape',
            robj = rev(x = dims),
            dtype = GuessDType(x = dims)
          )
          layers[[other]]$attr_delete(attr_name = 'dims')
        }
        AddEncoding(dname = paste('layers', other, sep = '/'))
        layer.features <- switch(
          EXPR = layer.slot,
          'scale.data' = 'scaled.features',
          'features'
        )
        var.name <- paste0(other, '_features')
        dfile[['var']]$obj_copy_from(
          src_loc = source[['assays']][[other]],
          src_name = layer.features,
          dst_name = var.name
        )
        col.order <- h5attr(x = dfile[['var']], which = 'column-order')
        col.order <- c(col.order, var.name)
        dfile[['var']]$attr_rename(
          old_attr_name = 'column-order',
          new_attr_name = 'old-column-order'
        )
        dfile[['var']]$create_attr(
          attr_name = 'column-order',
          robj = col.order,
          dtype = GuessDType(x = col.order)
        )
        dfile[['var']]$attr_delete(attr_name = 'old-column-order')
      }
    }
    if (!length(x = names(x = layers))) {
      dfile$link_delete(name = 'layers')
    }
  }
  # Because AnnData can't handle an empty /uns
  if (!length(x = names(x = dfile[['uns']]))) {
    dfile$link_delete(name = 'uns')
  }
  dfile$flush()
  return(dfile)
}

t1=proc.time()
if ( is.null(infile) ) {
  print('positional argument `infile` is null')
  quit('no', -1)
}

filename = str_split(basename(infile),".h5ad$",simplify=T)[,1]
Convert(infile, dest = paste0('./',filename,'.h5seurat'),assay = assay, overwrite = TRUE)

f <- H5File$new(paste0('./',filename,'.h5seurat'), "r+")
groups <- f$ls(recursive = TRUE)

for (name in groups$name[grepl("(?i)(?=.*meta.data.*)(?=.*categories.*).*",groups$name,perl=TRUE)]) {
  names <- strsplit(name, "/")[[1]]
  names <- c(names[1:length(names) - 1], "levels")
  new_name <- paste(names, collapse = "/")
  f[[new_name]] <- f[[name]]
}

for (name in groups$name[grepl("(?i)(?=.*meta.data.*)(?=.*code.*).*",groups$name,perl=TRUE)]) {
  names <- strsplit(name, "/")[[1]]
  names <- c(names[1:length(names) - 1], "values")
  new_name <- paste(names, collapse = "/")
  f[[new_name]] <- f[[name]]
  grp <- f[[new_name]]
  grp$write(args = list(1:grp$dims), value = grp$read() + 1)
}

f$close_all()

h5_file <- paste0('./',filename,'.h5seurat')
print(paste(c("Finished! Converting h5ad to h5seurat file at:", h5_file), sep=" ", collapse=NULL))
t2=proc.time()
t=t2-t1
print(paste0('h5seurat time consuming: ',t[3][[1]]))

object <- LoadH5Seurat(h5_file, misc=F)
print(paste(c("Successfully load h5seurat:", h5_file), sep=" ", collapse=NULL))

library(rhdf5)
object@misc <- h5read(h5_file,"/misc/")
detach("package:rhdf5", unload=TRUE)

if (
    !is.null(object@misc$sct_counts) &&
    !is.null(object@misc$sct_data) &&
    !is.null(object@misc$sct_scale) &&
    !is.null(object@misc$sct_cellname) &&
    !is.null(object@misc$sct_genename) &&
    !is.null(object@misc$sct_top_features)
  ) {
  sct.assay.out <- CreateAssayObject(counts=object[[assay]]@counts, check.matrix=FALSE)
  sct.assay.out <- SetAssayData(
      object = sct.assay.out,
      slot = "data",
      new.data = log1p(x=GetAssayData(object=sct.assay.out, slot="counts"))
    )
  sct.assay.out@scale.data <- object[[assay]]@scale.data
  colnames(sct.assay.out@scale.data) <- object@misc$sct_cellname
  rownames(sct.assay.out@scale.data) <- object@misc$sct_scale_genename
  sct.assay.out <- Seurat:::SCTAssay(sct.assay.out, assay.orig=assay)
  Seurat::VariableFeatures(object = sct.assay.out) <- object@misc$sct_top_features
  object[['SCT']] <- sct.assay.out
  DefaultAssay(object=object) <- 'SCT'

  # TODO: tag the reductions as SCT, this will influence the find_cluster choice of data
  if (!is.null(object@reductions$pca) && !is.null(object@reductions$umap)) {
    object@reductions$pca@assay.used <- 'SCT'
    object@reductions$umap@assay.used <- 'SCT'
  }
  assay.used <- 'SCT'
  print("Finished! Got SCTransform result in object, create a new SCTAssay and set it as default assay.")
} else {
  # TODO: we now only save raw counts, try not to add raw counts to .data, do NormalizeData to fit this
  # object <- NormalizeData(object)
  assay.used <- assay
  # print("Finished! Got raw counts only, auto create log-normalize data.")
}

# spatial already transform to `Spatial`` in assays
#if (!is.null(object@reductions$spatial)) {
#  object@reductions$spatial <- NULL
#}

# TODO follow with old code, don't touch
# add image
#if (
#    !is.null(object@meta.data$x) &&
#    !is.null(object@meta.data$y)
#   ){
#    print("Start add image...This may take some minutes...(~.~)")
#    cell_coords <- unique(object@meta.data[, c('x', 'y')])
#    cell_coords['cell'] <- row.names(cell_coords)
#    cell_coords$x <- cell_coords$x - min(cell_coords$x) + 1
#    cell_coords$y <- cell_coords$y - min(cell_coords$y) + 1
#
#    # object of images$slice1@image, all illustrated as 1 since no concrete pic
#    tissue_lowres_image <- matrix(1, max(cell_coords$y), max(cell_coords$x))
#
#    # object of images$slice1@coordinates, concrete coordinate of X and Y
#    tissue_positions_list <- data.frame(row.names = cell_coords$cell,
#                                    tissue = 1,
#                                    row = cell_coords$y, col = cell_coords$x,
#                                    imagerow = cell_coords$y, imagecol = cell_coords$x)
#    # @images$slice1@scale.factors
#    scalefactors_json <- toJSON(list(fiducial_diameter_fullres = 1, tissue_hires_scalef = 1, tissue_lowres_scalef = 1))
#
#    # generate object @images$slice1
#    generate_BGI_spatial <- function(image, scale.factors, tissue.positions, filter.matrix = TRUE) {
#    if (filter.matrix) {
#      tissue.positions <- tissue.positions[which(tissue.positions$tissue == 1), , drop = FALSE]
#    }
#    unnormalized.radius <- scale.factors$fiducial_diameter_fullres * scale.factors$tissue_lowres_scalef
#    spot.radius <- unnormalized.radius / max(dim(x = image))
#    return(new(Class = 'VisiumV1',
#             image = image,
#             scale.factors = scalefactors(spot = scale.factors$tissue_hires_scalef,
#                                          fiducial = scale.factors$fiducial_diameter_fullres,
#                                          hires = scale.factors$tissue_hires_scalef,
#                                          lowres = scale.factors$tissue_lowres_scalef),
#             coordinates = tissue.positions,
#             spot.radius = spot.radius))
#    }
#
#    BGI_spatial <- generate_BGI_spatial(image = tissue_lowres_image,
#                                    scale.factors = fromJSON(scalefactors_json),
#                                    tissue.positions = tissue_positions_list)
#
#    # can be thought of as a background of spatial
#    # import image into seurat object
#    assay.used <- 'Spatial'
#    object@images[['slice1']] <- BGI_spatial
#    object@images$slice1@key <- "slice1_"
#    object@images$slice1@assay <- assay.used
#}

t3=proc.time()
t=t3-t1
print(paste0('rds time consuming: ',t[3][[1]]))

print("Start to saveRDS...")
if (outfile=='None'){
    filename = str_split(basename(infile),".h5ad$",simplify=T)[,1]
}else{
    filename = outfile
}
# outfile = paste0('./',filename)
saveRDS(object, outfile)
print("Finished RDS.")
quit('yes', 0)
