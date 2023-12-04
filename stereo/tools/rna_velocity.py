# @FileName : rna_velocity.py
# @Time     : 2022-09-19 10:12:00
# @Author   : Xujunhao
# @Email    : xujunhao@genomics.cn
import os
import time

import gtfparse as gp
import loompy
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from shapely.geometry import MultiPoint
from shapely.geometry import Point
from tqdm import tqdm

from ..log_manager import logger


class RnaVelocity(object):

    def __init__(self, gem_path=None, gef_path=None, gtf_path=None, out_dir=None, bin_size=100):
        self.gem_path = gem_path
        self.gef_path = gef_path
        self.gtf_path = gtf_path
        self.out_dir = out_dir
        self.bin_size = bin_size
        self.check_input()

    def check_input(self):
        if self.gef_path is None and self.gem_path is None:
            logger.info("must to input gem file or gef file")
            raise Exception("must to input gem file or gef file")

        if self.gtf_path is None:
            logger.info("must to input gtf file")
            raise Exception("must to input gtf file")

        if self.out_dir is None:
            now = time.strftime("%Y%m%d%H%M%S")
            self.out_dir = f"./rna_velocity_result_{now}"
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def check_gem_exon(self, df, key):
        if key in list(df.columns.values):
            return True
        else:
            return False

    def generate_loom_with_square_bin(self):
        if self.gef_path is None and self.gem_path is not None:
            logger.info("Getting layers")
            df = pd.read_csv(str(self.gem_path), sep='\t', comment='#', header=0)
            flag = self.check_gem_exon(df, 'ExonCount')
            if not flag:
                logger.info("Exon information not found in gem file")
                raise Exception("Exon information not found in gem file")
            df = self.colchange(df)
            df_exp_bin = self.parse_bin_coor(df, self.bin_size)
            layers = self.get_layers(df_exp_bin)

            logger.info("Getting row attrs from gtf")
            df_row_attrs = self.get_row_attrs(layers["total"])

            logger.info("Generating loom")
            output_loom_file = self.loom_generate(df_row_attrs, layers)

        else:
            logger.info("Getting layers")
            from gefpy.bgef_reader_cy import BgefR
            gef = BgefR(self.gef_path, self.bin_size, 4)
            # Determine whether the gef file contains exon information
            if gef.is_Contain_Exon():
                # do not do any filtering
                region = []
                gene_list = []
                uniq_cell, gene_names, count, cell_ind, gene_ind, exon = gef.get_filtered_data_exon(region, gene_list)

                layers = self.get_layers_gef(uniq_cell, gene_names, count, cell_ind, gene_ind, exon)
                logger.info("Getting row attrs from gtf")
                df_row_attrs = self.get_row_attrs(layers["total"])

                logger.info("Generating loom")
                output_loom_file = self.loom_generate(df_row_attrs, layers)

            else:
                logger.info("Exon information not found in gef file")
                raise Exception("Exon information not found in gef file")

        return output_loom_file

    def generate_loom_with_cell_bin(self):
        if self.gef_path is None and self.gem_path is not None:
            logger.info("Getting layers")
            df = pd.read_csv(str(self.gem_path), sep='\t', comment='#', header=0)
            flag = self.check_gem_exon(df, 'ExonCount')
            if not flag:
                logger.info("Exon information not found in gem file")
                raise Exception("Exon information not found in gem file")
            df = self.colchange(df)
            df_exp_bin = self.parse_cell_bin_coor(df)
            layers = self.get_layers(df_exp_bin)

            logger.info("Getting row attrs from gtf")
            df_row_attrs = self.get_row_attrs(layers["total"])

            logger.info("Generating loom")
            output_loom_file = self.loom_generate(df_row_attrs, layers)

        else:
            logger.info("Getting layers")
            from gefpy.cgef_reader_cy import CgefR
            gef = CgefR(self.gef_path)
            # Determine whether the gef file contains exon information
            if gef.is_Contain_Exon():
                # do not do any filtering
                region = []
                gene_list = []
                uniq_cell, gene_names, count, cell_ind, gene_ind, exon = gef.get_filtered_data_exon(region, gene_list)

                layers = self.get_layers_gef(uniq_cell, gene_names, count, cell_ind, gene_ind, exon)
                logger.info("Getting row attrs from gtf")
                df_row_attrs = self.get_row_attrs(layers["total"])

                logger.info("Generating loom")
                output_loom_file = self.loom_generate(df_row_attrs, layers)

            else:
                logger.info("Exon information not found in gef file")
                raise Exception("Exon information not found in gem file")

        return output_loom_file

    def parse_cell_bin_coor(self, df):
        gdf = df.groupby('CellID').apply(lambda x: self.make_multipoint(x))
        df = pd.merge(df, gdf, on='CellID')
        exp_bin = df.groupby(["cell_id", "geneID"]).agg({"MIDCount": "sum", "ExonCount": "sum"})
        return exp_bin

    def make_multipoint(self, x):
        p = [Point(i) for i in zip(x['x'], x['y'])]
        mlp = MultiPoint(p).convex_hull
        x_center = round(mlp.centroid.x, 4)
        y_center = round(mlp.centroid.y, 4)
        cell_id = str(x_center) + '_' + str(y_center)
        return pd.Series({'cell_id': cell_id})

    def colchange(self, df):
        if "MIDCounts" in df.columns:
            df = df.rename(columns={"MIDCounts": "MIDCount"})
        elif "UMICounts" in df.columns:
            df = df.rename(columns={"UMICounts": "MIDCount"})
        elif "UMICount" in df.columns:
            df = df.rename(columns={"UMICount": "MIDCount"})

        if 'label' in df.columns:
            df.rename(columns={'label': 'CellID'}, inplace=True)

        df.dropna(inplace=True)
        return df

    def parse_bin_coor(self, df, bin_size):
        """
        merge bins to a bin unit according to the bin size,
        and generate cell id of bin unit using the coordinate after merged.

        :param df: a dataframe of the bin file.
        :param bin_size: the size of bin to merge.

        :return: a dataframe
        """
        x_min = df['x'].min()
        y_min = df['y'].min()
        df['bin_x'] = self.merge_bin_coor(df['x'].values, x_min, bin_size)
        df['bin_y'] = self.merge_bin_coor(df['y'].values, y_min, bin_size)
        df['cell_id'] = df['bin_x'].astype(str) + '_' + df['bin_y'].astype(str)
        exp_bin = df.groupby(["cell_id", "geneID"]).agg({"MIDCount": "sum", "ExonCount": "sum"})

        return exp_bin

    def merge_bin_coor(self, coor: np.ndarray, coor_min: int, bin_size: int):
        return np.floor((coor - coor_min) / bin_size).astype(np.int)

    def get_layers(self, df_exp_bin):
        """
        generate total_count, extron, intron matrix information according gem file.

        :df_exp_bin: dataframe of gene expression.

        :return: dictionnary of total_count, extron, intron matrix information.
        """
        layer_total = self.cal_layer(df_exp_bin, "matrix")
        layer_extron = self.cal_layer(df_exp_bin, "extron")
        layer_intron = layer_total - layer_extron
        layer_ambiguous = layer_total - layer_extron - layer_intron

        return {"total": layer_total, "extron": layer_extron, "intron": layer_intron, "ambiguous": layer_ambiguous}

    def get_layers_gef(self, uniq_cell, gene_names, count, cell_ind, gene_ind, exon):
        """
        generate total_count, extron, intron matrix information according gef file.

        :uniq_cell: cell id.
        :gene_names: gene_id.
        :count: list of total expression count.
        :cell_ind: list of cell index.
        :gene_ind: list of gene index.
        :exon: list of extron expression count.

        :return: dictionnary of total_count, extron, intron matrix information
        """

        x_array = np.right_shift(uniq_cell, 32).astype('str')
        y_array = np.bitwise_and(uniq_cell, 0xffffffff).astype('str')
        position_cell_name = ['_'.join(map(str, i)) for i in zip(x_array, y_array)]
        cn = len(uniq_cell)
        gn = len(gene_names)
        total_exp_matrix_coo = sparse.coo_matrix((count, (gene_ind, cell_ind)), shape=(gn, cn), dtype=np.uint32)
        layer_total = pd.DataFrame(total_exp_matrix_coo.toarray(), columns=position_cell_name, index=gene_names)

        eson_exp_matrix_coo = sparse.coo_matrix((exon, (gene_ind, cell_ind)), shape=(gn, cn), dtype=np.uint32)
        layer_extron = pd.DataFrame(eson_exp_matrix_coo.toarray(), columns=position_cell_name, index=gene_names)
        layer_intron = layer_total - layer_extron
        layer_ambiguous = layer_total - layer_extron - layer_intron

        return {"total": layer_total, "extron": layer_extron, "intron": layer_intron, "ambiguous": layer_ambiguous}

    def cal_layer(self, df, which):
        """
        calculate sum value of MIDCount and ExonCount

        :df: dataframe of gene expression.
        :which: matrix or extron.

        :return: dataframe summed based on geneID and cellID.
        """

        df = df.reset_index()
        bin_cell = list(set(df["geneID"]))
        chunks = [bin_cell[x:x + 5000] for x in range(0, len(bin_cell), 5000)]
        chunks_exp = []
        if which == "matrix":
            for i in tqdm(chunks):
                chunks_exp.append(
                    pd.pivot(df[df["geneID"].isin(i)], columns="cell_id", index="geneID", values='MIDCount'))
        elif which == "extron":
            for i in tqdm(chunks):
                chunks_exp.append(
                    pd.pivot(df[df["geneID"].isin(i)], columns="cell_id", index="geneID", values='ExonCount'))
        else:
            raise

        c_df = pd.concat(chunks_exp, join="outer").fillna(0)

        return c_df

    def get_row_attrs(self, layer_total):
        """
        get row attrs from gtf

        :layer_total: dataframe of total_count matrix(row: cell_id; col:gene_id).

        :return: dataframe of annotaion information.
        """
        base = gp.read_gtf(self.gtf_path)
        gene_list = list(layer_total.index)

        base_sub = base.loc[
            base["feature"] == "gene", ["gene_id", "gene_name", "seqname", "strand", "start", "end"]].copy()
        base_sub.drop_duplicates(keep="first", inplace=True)

        base_need = base_sub.loc[np.isin(base_sub["gene_name"], gene_list), :].copy()
        base_need.drop_duplicates(subset=['gene_name'], keep='first', inplace=True)

        df_row_attrs = pd.DataFrame(index=gene_list)
        df_row_attrs = df_row_attrs.reset_index()
        df_row_attrs.rename(columns={"index": "gene_name"}, inplace=True)
        df_row_attrs = df_row_attrs.merge(base_need, on="gene_name", how="left")

        return df_row_attrs

    def loom_generate(self, df_row_attrs, layers):
        """
        generate loom file by using loompy

        :df_row_attrs: dataframe of annotaion information.
        :layers: total_count, extron, intron matrix information.

        :return: the output loom path.
        """
        row_attrs = {"Accession": np.array(df_row_attrs.gene_id),
                     "Chromosome": np.array(df_row_attrs.seqname),
                     "End": np.array(df_row_attrs.end),
                     "Gene": np.array(df_row_attrs.gene_name),
                     "Start": np.array(df_row_attrs.start),
                     "Strand": np.array(df_row_attrs.strand)}

        col_attrs = {"CellID": np.array(layers["total"].columns)}

        file_out = os.path.join(self.out_dir, "rna_velocity.loom")

        loompy.create(file_out, sparse.coo_matrix(np.array(layers["total"])), row_attrs=row_attrs, col_attrs=col_attrs)

        with loompy.connect(file_out) as ds:
            ds.layers['spliced'] = sparse.coo_matrix(np.array(layers["extron"]), dtype='uint32')
            ds.layers['unspliced'] = sparse.coo_matrix(np.array(layers["intron"]), dtype='uint32')
            ds.layers['ambiguous'] = sparse.coo_matrix(np.array(layers["ambiguous"]), dtype='uint32')

        return file_out


def generate_loom(out_dir=None,
                  gem_path=None,
                  gef_path=None,
                  gtf_path=None,
                  bin_type='bins',
                  bin_size=100):
    """
    generate loom file for Dynamo according gem/gef and gtf file

    :param out_dir: the output path.
    :param gem_path: the path of gem file, if None, need to input gef_path, defaults to None.
    :param gef_path: the path of gef file, if None, need to input gem_path and then generate from gem, defaults to None.
    :param gtf_path: the path of gtf file, if None, need to input gtf_path, defaults to None.
    :param bin_type: the type of bin, if file format is stereo-seq file. `bins` or `cell_bins`.
    :param bin_size: the size of bin to merge. The parameter only takes effect.
                    when the value of data.bin_type is 'bins'.

    :return: the output loom path.
    """

    rv = RnaVelocity(gem_path=gem_path, gef_path=gef_path, gtf_path=gtf_path, out_dir=out_dir, bin_size=bin_size)
    if bin_type == 'bins':
        output_loom_file = rv.generate_loom_with_square_bin()
        return output_loom_file
    elif bin_type == 'cell_bins':
        output_loom_file = rv.generate_loom_with_cell_bin()
        return output_loom_file
    else:
        logger.info("Please set cell_type to `bins` or `cell_bins`")
