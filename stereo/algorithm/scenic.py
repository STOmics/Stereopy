import glob
from arboreto.utils import load_tf_names
from arboreto.algo import grnboost2
from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
from pyscenic.utils import modules_from_adjacencies
from pyscenic.prune import prune2df, df2regulons
from pyscenic.aucell import aucell
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['NUMBA_THREADING_LAYER'] = 'omp'


def scenic(data, tfs, motif, database_dir, outdir=None,):
    """

    :param data: StereoExpData
    :param tfs: tfs file in txt format
    :param motif: motif file in tbl format
    :param database_dir: directory containing reference database(.feather files), cisTarget
    :param outdir: directory containing output files(including modules.pkl, regulons.csv, adjacencies.tsv, motifs.csv).
    If None, results will not be output to files.

    :return:
    """
    ex_matrix = data.to_df()
    tf_names = load_tf_names(tfs) # Derive list of Transcription Factors(TF) for Mus musculus
    db_fnames = glob.glob(os.path.join(database_dir,"*feather")) # Load ranking databases
    dbs = [RankingDatabase(fname=fname, name=get_name(fname)) for fname in db_fnames]
    # Phase I: Inference of co-expression modules
    # Run GRNboost from arboreto to infer co-expression modules
    adjacencies = grnboost2(expression_data=ex_matrix, tf_names=tf_names, verbose=True)
    # Derive potential regulomes from these co-expression modules
    modules = list(modules_from_adjacencies(adjacencies, ex_matrix))
    # Phase II: Prune modules for targets with cis regulatory footprints (aka RcisTarget)
    motifs = prune2df(dbs, modules, motif)
    regulons = df2regulons(motifs)
    regulons_df = pd.DataFrame.from_dict(regulons[0].gene2weight, orient="index", columns=["value"])
    regulons_df["TF"] = regulons[0].name
    for i in range(1, len(regulons)):
        temp = pd.DataFrame.from_dict(regulons[i].gene2weight, orient="index", columns=["value"])
        temp["TF"] = regulons[i].name
        regulons_df = pd.concat([regulons_df, temp])
    # Phase III: Cellular regulon enrichment matrix (aka AUCell)
    auc_mtx = aucell(ex_matrix, regulons, num_workers=1)
    #sns.clustermap(auc_mtx, figsize=(12, 12))
    if outdir is not None:
        from stereo.io.writer import save_pkl
        save_pkl(modules,output=f"{outdir}/modules.pkl")
        regulons_df.to_csv(f"{outdir}/regulons_gene2weight.csv")
        save_pkl(regulons,output=f"{outdir}/regulons.pkl")
        adjacencies.to_csv(f"{outdir}/adjacencies.tsv")
        motifs.to_csv(f"{outdir}/motifs.csv")
        auc_mtx.to_csv(f"{outdir}/aux.csv")
    return modules, regulons, adjacencies, motifs, auc_mtx, regulons_df


def get_name(fname):
    return os.path.splitext(os.path.basename(fname))[0]