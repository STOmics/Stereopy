import json
import math
import os
import re
import time
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from io import StringIO
from copy import deepcopy

import numpy as np
import pandas as pd
from IPython.display import IFrame
from natsort import natsorted
from scipy import stats

from stereo.core.stereo_exp_data import StereoExpData
from stereo.preprocess import filter_genes
from .PAGA_traj import cal_plt_param_traj_clus_from_adata


# import anndata as ad

def updateItem(item):
    if isinstance(item, str):
        return re.sub("[^a-zA-Z0-9-]", '_', item)
    if isinstance(item, int) or isinstance(item, float) or isinstance(item, np.int64) or isinstance(
            item, np.int32) or isinstance(item, np.float32) or isinstance(item, np.float64):
        if math.isnan(item):
            return 'NA'
        else:
            return f'{int(item)}'


def UpdateList(xxxarr, return_ndarray=False):
    tmp = pd.DataFrame()
    tmp['v1'] = list(xxxarr)
    tmp['v2'] = tmp.apply(lambda row: updateItem(row['v1']), axis=1)
    return tmp['v2'].to_list() if not return_ndarray else tmp['v2'].to_numpy()


class my_json_encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64) or isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.float64) or isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def getPAGACurves(data, ty_col='annotation', choose_ty=None, trim=True, paga_key='paga', mesh_key='mesh'):
    parameters_data = cal_plt_param_traj_clus_from_adata(data, ty_col=ty_col, choose_ty=choose_ty, trim=trim,
                                                         type_traj='curve', paga_key=paga_key, mesh_key=mesh_key)
    x_unknown_li_all_tra, y_unknown_li_all_tra, z_unknown_li_all_tra, com_tra_li, com_tra_wei_li = parameters_data
    traj_names = []
    traj_lines = []
    traj_widths = []
    for i, sin_tra in enumerate(com_tra_li):  # 对每条完整的轨迹
        for j in range(len(sin_tra) - 1):  # 对于这条轨迹每一个截断
            traj_name = f'{sin_tra[j]}_{sin_tra[j + 1]}'
            traj_names.append(traj_name)
            traj_x = x_unknown_li_all_tra[i][j].tolist()
            traj_y = y_unknown_li_all_tra[i][j].tolist()
            traj_z = z_unknown_li_all_tra[i][j].tolist()
            traj_line = []
            for m in range(len(traj_x)):
                traj_line.append([traj_x[m], traj_y[m], traj_z[m]])
            traj_lines.append(traj_line)
            traj_W = com_tra_wei_li[i][j]
            traj_widths.append(traj_W)
    return [traj_names, traj_lines, traj_widths]


def getPAGALines(data, ty_col='annotation', choose_ty=None, trim=True, paga_key='paga', mesh_key='mesh'):
    parameters_data = cal_plt_param_traj_clus_from_adata(data, ty_col=ty_col, choose_ty=choose_ty, trim=trim,
                                                         type_traj='line', paga_key=paga_key, mesh_key=mesh_key)
    x_unknown_li_all_tra, y_unknown_li_all_tra, z_unknown_li_all_tra, com_tra_li, com_tra_wei_li = parameters_data

    traj_names = []
    traj_lines = []
    traj_widths = []
    for i, tnames in enumerate(com_tra_li):  # 对每条完整的轨迹
        traj_name = f'{tnames[0]}_{tnames[1]}'
        traj_names.append(traj_name)
        traj_line = [[x_unknown_li_all_tra[i][0], y_unknown_li_all_tra[i][0], z_unknown_li_all_tra[i][0]],
                     [x_unknown_li_all_tra[i][1], y_unknown_li_all_tra[i][1], z_unknown_li_all_tra[i][1]]]
        traj_lines.append(traj_line)
        traj_W = com_tra_wei_li[i]
        traj_widths.append(traj_W)
    return [traj_names, traj_lines, traj_widths]


class Meshes:
    def __init__(self):
        # meshname, vectors, faces
        self._data = [[], [], []]

    @property
    def data(self):
        return self._data

    @staticmethod
    def MesheStr(objfile):
        mesh_str = ''
        file1 = open(objfile, 'r')
        Lines = file1.readlines()
        file1.close()
        for line in Lines:
            if len(line) > 1 and (line[0] == 'v' or line[0] == 'f'):
                mesh_str = mesh_str + line
        return mesh_str

    def add_mesh(self, meshname, objfile):
        if isinstance(objfile, str):
            self._add_mesh_str(meshname, objfile)
        elif isinstance(objfile, dict):
            self._add_mesh_dict(meshname, objfile)

    def _add_mesh_dict(self, meshname, dict_info):
        vectors = pd.DataFrame()
        vectors['x'] = dict_info['points'][:, 0]
        vectors['y'] = dict_info['points'][:, 1]
        vectors['z'] = dict_info['points'][:, 2]
        vectors = vectors.astype(float)
        xmin = vectors['x'].min()
        xmax = vectors['x'].max()
        ymin = vectors['y'].min()
        ymax = vectors['y'].max()
        zmin = vectors['z'].min()
        zmax = vectors['z'].max()
        if len(self._data[0]) == 0:
            self.mesh_xmin = xmin
            self.mesh_xmax = xmax
            self.mesh_ymin = ymin
            self.mesh_ymax = ymax
            self.mesh_zmin = zmin
            self.mesh_zmax = zmax
        else:
            if xmin < self.mesh_xmin:
                self.mesh_xmin = xmin
            if ymin < self.mesh_ymin:
                self.mesh_ymin = ymin
            if zmin < self.mesh_zmin:
                self.mesh_zmin = zmin
            if xmax > self.mesh_xmax:
                self.mesh_xmax = xmax
            if ymax > self.mesh_ymax:
                self.mesh_ymax = ymax
            if zmax > self.mesh_zmax:
                self.mesh_zmax = zmax

        faces = []
        for four_points in dict_info['faces']:
            if four_points[0] == 3:
                faces.append([four_points[1], four_points[2], four_points[3]])
            elif four_points[0] == 4:
                faces.append([four_points[1], four_points[2], four_points[3]])
                faces.append([four_points[1], four_points[3], four_points[4]])
            else:
                print('Error: not triangle or quadrilateral mesh')
                return
        self._data[0].append(meshname)
        self._data[1].append(vectors.to_numpy().tolist())
        self._data[2].append(faces)

    def _add_mesh_str(self, meshname, objfile):
        mesh_io = StringIO(Meshes.MesheStr(objfile))
        cache = pd.read_csv(mesh_io, sep='\s+', header=None, compression='infer', comment='#')  # noqa
        cache.columns = ['type', 'v1', 'v2', 'v3']
        vectors = cache[cache['type'] == 'v'].copy()
        vectors = vectors[['v1', 'v2', 'v3']].copy()
        vectors.columns = ['x', 'y', 'z']
        vectors = vectors.astype(float)
        xmin = vectors['x'].min()
        xmax = vectors['x'].max()
        ymin = vectors['y'].min()
        ymax = vectors['y'].max()
        zmin = vectors['z'].min()
        zmax = vectors['z'].max()
        if len(self._data[0]) == 0:
            self.mesh_xmin = xmin
            self.mesh_xmax = xmax
            self.mesh_ymin = ymin
            self.mesh_ymax = ymax
            self.mesh_zmin = zmin
            self.mesh_zmax = zmax
        else:
            if xmin < self.mesh_xmin:
                self.mesh_xmin = xmin
            if ymin < self.mesh_ymin:
                self.mesh_ymin = ymin
            if zmin < self.mesh_zmin:
                self.mesh_zmin = zmin
            if xmax > self.mesh_xmax:
                self.mesh_xmax = xmax
            if ymax > self.mesh_ymax:
                self.mesh_ymax = ymax
            if zmax > self.mesh_zmax:
                self.mesh_zmax = zmax
        self._data[0].append(meshname)
        self._data[1].append(vectors.to_numpy().tolist())

        faces = cache[cache['type'] == 'f'].copy()
        if faces.dtypes['v1'] == object:
            faces['i'] = faces.apply(lambda row: int(row['v1'].split('/')[0]) - 1, axis=1)
            faces['j'] = faces.apply(lambda row: int(row['v2'].split('/')[0]) - 1, axis=1)
            faces['k'] = faces.apply(lambda row: int(row['v3'].split('/')[0]) - 1, axis=1)
        else:
            faces['i'] = faces['v1'] - 1
            faces['j'] = faces['v2'] - 1
            faces['k'] = faces['v3'] - 1
        faces = faces[['i', 'j', 'k']].copy()
        self._data[2].append(faces.to_numpy().tolist())

    def update_summary(self, summary):
        ret = summary
        if self.mesh_xmin < ret['box']['xmin']:
            ret['box']['xmin'] = self.mesh_xmin
        if self.mesh_xmax > ret['box']['xmax']:
            ret['box']['xmax'] = self.mesh_xmax
        if self.mesh_ymin < ret['box']['ymin']:
            ret['box']['ymin'] = self.mesh_ymin
        if self.mesh_ymax > ret['box']['ymax']:
            ret['box']['ymax'] = self.mesh_ymax
        if self.mesh_zmin < ret['box']['zmin']:
            ret['box']['zmin'] = self.mesh_zmin
        if self.mesh_zmax > ret['box']['zmax']:
            ret['box']['zmax'] = self.mesh_zmax
        return ret


class Stereo3DWebCache:
    """
    Analyse the 3D SRT data and provide detailed json data for the data browser.
    """

    def __init__(
            self,
            data: StereoExpData,
            meshes={},
            cluster_label=None,
            spatial_label='spatial_rigid',
            exp_cutoff=0,
            paga_key='paga',
            grn_key='regulatory_network_inference',
            ccc_key='cell_cell_communication'
    ):
        self._data = data
        self.__subdata = None
        self._cluster_label = cluster_label
        self._spatkey = spatial_label
        self._grn_key = grn_key
        self._paga_key = paga_key
        self._ccc_key = ccc_key
        self._expcutoff = exp_cutoff
        self._init_atlas_summary()
        self._init_meshes(meshes)
        self._update_atlas_summary()

    def _init_atlas_summary(self):
        """
        get summary dic of atlas
        """
        self._summary = {}
        # get the total xxx
        self._summary['total_cell'] = len(self._data.cell_names)
        self._summary['total_gene'] = len(self._data.gene_names)
        # get Annotation factors
        if self._cluster_label is not None:
            self._summary['annokeys'] = [self._cluster_label]
            self._summary['annomapper'] = {}
            # get Annotation labels
            if self._cluster_label in self._data.cells._obs.columns:
                unique_anno = natsorted(np.unique(self._data.cells._obs[self._cluster_label]))
            else:
                unique_anno = np.unique(self._data.tl.result[self._cluster_label]['group'])
            legend2int = {}
            int2legend = {}
            for i, key in enumerate(unique_anno):
                legend2int[key] = i
                int2legend[i] = key
            self._summary['annomapper'][f'{self._cluster_label}_legend2int'] = legend2int
            self._summary['annomapper'][f'{self._cluster_label}_int2legend'] = int2legend
            # prepare box-space
        self._summary['box'] = {}
        self._summary['box']['xmin'] = np.min(self._data.position[:, 0])
        self._summary['box']['xmax'] = np.max(self._data.position[:, 0])
        self._summary['box']['ymin'] = np.min(self._data.position[:, 1])
        self._summary['box']['ymax'] = np.max(self._data.position[:, 1])
        self._summary['box']['zmin'] = np.min(self._data.position_z)
        self._summary['box']['zmax'] = np.max(self._data.position_z)

        self._summary['option'] = {
            'default': 'CellTypes',
            "CellTypes": False,
            "GeneExpression": True,
            "Digital_in_situ": True,
            'PAGA_trajectory': False,
            'GRN_Regulons': False,
            'Hotspot_Modules': False,
            'Cell_Cell_Communication': False,
        }
        if self._cluster_label is not None:
            self._summary['option']['CellTypes'] = True
        if (self._paga_key is not None) and (self._paga_key in self._data.tl.result):
            self._summary['option']['PAGA_trajectory'] = True
            self._summary['option']['default'] = 'PAGA_trajectory'
        if (self._ccc_key is not None) and (self._ccc_key in self._data.tl.result):
            self._summary['option']['Cell_Cell_Communication'] = True
            self._summary['option']['default'] = 'Cell_Cell_Communication'
            self._data.tl.result[self._ccc_key]['visualization_data']['celltype1'] = UpdateList(
                self._data.tl.result[self._ccc_key]['visualization_data']['celltype1'])
            self._data.tl.result[self._ccc_key]['visualization_data']['celltype2'] = UpdateList(
                self._data.tl.result[self._ccc_key]['visualization_data']['celltype2'])
            self._data.tl.result[self._ccc_key]['visualization_data']['ligand'] = UpdateList(
                self._data.tl.result[self._ccc_key]['visualization_data']['ligand'])
            self._data.tl.result[self._ccc_key]['visualization_data']['receptor'] = UpdateList(
                self._data.tl.result[self._ccc_key]['visualization_data']['receptor'])
        if (self._grn_key is not None) and (self._grn_key in self._data.tl.result):
            self._summary['option']['GRN_Regulons'] = True
            self._summary['option']['default'] = 'GRN_Regulons'
            self._data.tl.result[self._grn_key]['auc_matrix'].columns = UpdateList(
                self._data.tl.result[self._grn_key]['auc_matrix'].columns)

    def _init_meshes(self, meshes):
        """
        load all meshes
        """
        if len(meshes) > 0:
            self._has_mesh = True
            self._meshes = Meshes()
            for meshname in meshes:
                self._meshes.add_mesh(meshname, meshes[meshname])
        else:
            self._has_mesh = False

    def _update_atlas_summary(self):
        """
        update summary in case mesh is much bigger than scatter matrix
        """
        if self._has_mesh:
            self._summary = self._meshes.update_summary(self._summary)

    def get_summary(self):
        """
        return the summary.json
        """
        return json.dumps(self._summary, cls=my_json_encoder)

    def get_regulonnames(self):
        """
        return the regulon.json
        """
        return json.dumps(list(self._data.tl.result[self._grn_key]['auc_matrix'].columns), cls=my_json_encoder)

    def get_regulon(self, regulonname):
        """
        return the regulon.json
        """
        auc_mtx = self._data.tl.result[self._grn_key]['auc_matrix']
        if self.__subdata is None:
            self.__subdata = self._data.sub_by_name(cell_name=auc_mtx.index.to_numpy())
        xyz = np.concatenate([self.__subdata.position, self.__subdata.position_z], axis=1)
        sub_zscore = auc_mtx[regulonname]
        df = pd.DataFrame(data=xyz, columns=['x', 'y', 'z'])
        df['zscore'] = stats.zscore(sub_zscore.to_numpy())
        return json.dumps(df.to_numpy().tolist(), cls=my_json_encoder)

    def get_ccc_dict(self):
        """
        return the ccc_dict.json
        """
        ccc_df: pd.DataFrame = self._data.tl.result[self._ccc_key]['visualization_data']
        senders = ccc_df['celltype1'].unique()
        ret_dict = {}
        for sender in senders:
            ret_dict[sender] = {}
            subdf = ccc_df[ccc_df['celltype1'] == sender]
            recivers = subdf['celltype2'].unique()
            for reciver in recivers:
                ret_dict[sender][reciver] = {}
                subsubdf = subdf[subdf['celltype2'] == reciver]
                ligands = subsubdf['ligand'].unique()
                for ligand in ligands:
                    ret_dict[sender][reciver][ligand] = []
                    subsubsubdf = subsubdf[subsubdf['ligand'] == ligand]
                    receptors = subsubsubdf['receptor'].unique()
                    for receptor in receptors:
                        ret_dict[sender][reciver][ligand].append(receptor)
        return json.dumps(ret_dict, cls=my_json_encoder)

    def get_ccc_ct_gene(self, celltype, genename):
        if self._cluster_label in self._data.cells._obs.columns:
            is_in_bool = self._data.cells._obs[self._cluster_label].isin([celltype])
        else:
            is_in_bool = self._data.tl.result[self._cluster_label]['group'].isin([celltype])
        cell_names = self._data.cell_names[is_in_bool]
        subdata: StereoExpData = self._data.sub_by_name(cell_name=cell_names, gene_name=[genename])
        xyz = np.concatenate([subdata.position, subdata.position_z], axis=1)
        df = pd.DataFrame(data=xyz, columns=['x', 'y', 'z'])
        df['exp'] = subdata.exp_matrix.toarray() if subdata.issparse() else subdata.exp_matrix
        df = df[df['exp'] > self._expcutoff].copy()
        return json.dumps(df.to_numpy().tolist(), cls=my_json_encoder)

    def get_genenames(self):
        """
        return the gene.json
        """
        return json.dumps(self._data.gene_names.tolist(), cls=my_json_encoder)

    def get_gene(self, genename):
        """
        return the Gene/xxxgene.json
        """
        xyz = np.concatenate([self._data.position, self._data.position_z], axis=1)
        df = pd.DataFrame(data=xyz, columns=['x', 'y', 'z'])
        df = df.astype(int)  # force convert to int to save space
        genedata: StereoExpData = self._data.sub_by_name(gene_name=[genename])
        df['exp'] = genedata.exp_matrix.toarray() if genedata.issparse() else genedata.exp_matrix
        df = df[df['exp'] > self._expcutoff].copy()
        return json.dumps(df.to_numpy().tolist(), cls=my_json_encoder)

    def get_meshes(self):
        """
        return the meshes.json
        """
        if self._has_mesh:
            return json.dumps(self._meshes.data, cls=my_json_encoder)
        else:
            return ''

    def get_paga_line(self):
        """
        return the paga_line.json
        """
        return json.dumps(getPAGALines(self._data, ty_col=self._cluster_label, paga_key=self._paga_key))

    def get_paga(self):
        """
        return the paga.json
        """
        return json.dumps(getPAGACurves(self._data, ty_col=self._cluster_label, paga_key=self._paga_key))

    def get_anno(self):
        """
        return the Anno/xxxanno.json
        """
        xyz = np.concatenate([self._data.position, self._data.position_z], axis=1)
        df = pd.DataFrame(data=xyz, columns=['x', 'y', 'z'])
        # df = df.astype(int)  # force convert to int to save space
        if self._cluster_label in self._data.cells._obs.columns:
            df['anno'] = self._data.cells._obs[self._cluster_label].to_numpy()
        else:
            df['anno'] = self._data.tl.result[self._cluster_label]['group'].to_numpy()
        mapper = self._summary['annomapper'][f'{self._cluster_label}_legend2int']
        df['annoid'] = df.apply(lambda row: mapper[row['anno']], axis=1)
        return json.dumps(df[['x', 'y', 'z', 'annoid']].to_numpy().tolist(), cls=my_json_encoder)


class StoppableHTTPServer(HTTPServer):
    """
    The http server that stop when not_forever is called.
    """

    def serve_forever(self):
        self.stopped = False
        while not self.stopped:
            self.handle_request()
            time.sleep(0.100)

    def not_forever(self):
        print('Server terminate ...', flush=True)
        self.stopped = True
        self.server_close()


class ServerDataCache:
    """
    The template data cache.
    """

    def __init__(self):
        self._data_hook = None
        self._server = None
        self._front_dir = None

    @property
    def data_hook(self):
        return self._data_hook

    @property
    def server(self):
        return self._server

    @server.setter
    def server(self, http):
        self._server = http

    @data_hook.setter
    def data_hook(self, data_hook):
        self._data_hook = data_hook

    @property
    def front_dir(self):
        return self._front_dir

    @front_dir.setter
    def front_dir(self, dirname):
        self._front_dir = dirname


ServerInstance = ServerDataCache()


class DynamicRequstHander(BaseHTTPRequestHandler):
    """
    The request hander that return static browser files or detailed data jsons.
    """

    def _stop_server(self):
        ServerInstance.server.not_forever()
        self.send_response(404)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"Server shotdown now!")

    def _ret_static_files(self, the_relate_path, file_type):
        """
        return all static files like the browser codes and images
        """
        data_dir = ServerInstance.front_dir
        visit_path = f"{data_dir}{the_relate_path}"
        try:
            self.send_response(200)
            self.send_header('Content-type', file_type)
            self.end_headers()
            f = open(visit_path, 'rb')
            self.wfile.write(f.read())
            f.close()
        except Exception:
            self._ret_404()

    def _ret_jsonstr(self, jsonstr):
        if len(jsonstr) > 1:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(bytes(jsonstr, 'UTF-8'))

    def _ret_404(self):
        """
        return 404
        """
        self.send_response(404)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"404 Not Found")

    def do_GET(self):
        self.path = self.path.split('?')[0]
        if self.path in ['', '//', '/', '/index.html']:
            self._ret_static_files("/index.html", 'text/html')
        elif self.path == '/endnow':
            self._stop_server()
        elif self.path == '/summary.json':
            self._ret_jsonstr(ServerInstance.data_hook.get_summary())
        elif self.path == '/gene.json':
            self._ret_jsonstr(ServerInstance.data_hook.get_genenames())
        elif self.path == '/regulon.json':
            self._ret_jsonstr(ServerInstance.data_hook.get_regulonnames())
        elif self.path == '/ccc_dict.json':
            self._ret_jsonstr(ServerInstance.data_hook.get_ccc_dict())
        elif self.path == '/meshes.json':
            self._ret_jsonstr(ServerInstance.data_hook.get_meshes())
        elif self.path == '/paga.json':
            self._ret_jsonstr(ServerInstance.data_hook.get_paga())
        elif self.path == '/paga_line.json':
            self._ret_jsonstr(ServerInstance.data_hook.get_paga_line())
        elif self.path == '/test.json':  # handle json requst in the root path
            self._ret_jsonstr('{"test01":1.1, "test02":[1.1,3,2]}')
        elif self.path == '/conf.json':
            self._ret_jsonstr('')
        else:
            match_html = re.search('(.*).html$', self.path)
            match_js = re.search('(.*).js$', self.path)
            match_ttf = re.search('(.*).ttf$', self.path)
            match_API_regulon = re.search('/Regulon/(.*).json', self.path)
            match_API_gene = re.search('/Gene/(.*).json', self.path)
            match_API_anno = re.search('/Anno/(.*).json', self.path)
            match_API_CCC = re.search('/CCC/(.*)/(.*).json', self.path)
            match_API_scoexp = re.search('/gene_scoexp/(.*).json', self.path)
            if match_js:
                self._ret_static_files(self.path, 'application/javascript')
            elif match_html:
                self._ret_static_files(self.path, 'text/html')
            elif match_ttf:
                self._ret_static_files(self.path, 'application/x-font-ttf')
            elif match_API_regulon:
                regulon = match_API_regulon.group(1)
                self._ret_jsonstr(ServerInstance.data_hook.get_regulon(regulon))
            elif match_API_gene:
                genename = match_API_gene.group(1)
                self._ret_jsonstr(ServerInstance.data_hook.get_gene(genename))
            elif match_API_anno:
                self._ret_jsonstr(ServerInstance.data_hook.get_anno())
            elif match_API_CCC:
                celltype = match_API_CCC.group(1)
                genename = match_API_CCC.group(2)
                self._ret_jsonstr(ServerInstance.data_hook.get_ccc_ct_gene(celltype, genename))
            elif match_API_scoexp:
                self._ret_jsonstr('')
            else:
                self._ret_404()


def server_task(httpd, ip, port):
    # start endless waiting now...
    print('Server staring now ...')
    print(f'Call endServer("{ip}",{port}) to end this server')
    httpd.serve_forever()


def endServer(ip='127.0.0.1', port=7654):
    return IFrame(src=f'http://{ip}:{port}/endnow', width=500, height=50)


def launch(data: StereoExpData,
           cluster_label=None,
           spatial_label: str = 'spatial_rigid',
           meshes={},
           paga_key: str = None,
           grn_key: str = None,
           ccc_key: str = None,
           geneset=None,
           exp_cutoff=0,
           width=1600,
           height=1000,
           ip='127.0.0.1',
           port=7654,
           ):
    """
    Launch a data browser server based on input data

    :param datas: an AnnData object or a list of AnnData objects
    :param cluster_label: the keyword in obs for cluster/annotation info
    :param spatial_label: the keyword in obsm for 3D spatial coordinate
    :param mesh: all meshes in dict like, support obj or polydata (in numpy format) : {'heart': 'pathxxx/heart.obj', liver: adata.uns['mesh']['liver']}
    :param paga_key: paga data key in uns. if paga_key=None, or paga_key not exist in uns, the paga page will be disabled.
    :param grn_key: grn data key in uns. if grn_key=None, or grn_key not exist in uns, the grn page will be disabled.
    :param ccc_key: ccc data key in uns. if ccc_key=None, or ccc_key not exist in uns, the ccc page will be disabled.
    :param geneset: the specific geneset to display, show all genes in var if geneset is None
    :param exp_cutoff: the expression threshold to filter un-expression cells.
    :param width: the window width to render
    :param height: the window height to render
    :parma port: the port id

    :return:
    """  # noqa
    data = deepcopy(data)
    if paga_key is None and grn_key is None and ccc_key is None:
        raise Exception

    if grn_key is None:
        if paga_key is not None:
            cluster_label = data.tl.result[paga_key]['groups']
        else:
            cluster_label = data.tl.result[ccc_key]['parameters']['cluster_res_key']

    if cluster_label is not None:
        if cluster_label in data.cells._obs.columns:
            data.cells._obs[cluster_label] = UpdateList(data.cells._obs[cluster_label])
        else:
            data.tl.result[cluster_label]['group'] = UpdateList(data.tl.result[cluster_label]['group'])

    for meshname in meshes:
        meshfile = meshes[meshname]
        if isinstance(meshfile, str):
            if not os.path.isfile(meshfile):
                print(f'invalid obj :{meshfile}, return without any data browsing server...')
                return
        elif isinstance(meshfile, dict):
            continue
        else:
            print(f'invalid mesh data :{meshfile}, return without any data browsing server...')
    # filter by geneset now
    if geneset is not None:
        # data = data.sub_by_name(gene_name=geneset)  # notice, invalid gene will case program raising exceptions
        filter_genes(data, gene_list=geneset, inplace=True)
    data.genes.gene_name = UpdateList(data.genes.gene_name, return_ndarray=True).astype('U')
    # create core datacache
    datacache = Stereo3DWebCache(data, meshes, cluster_label, spatial_label, exp_cutoff, paga_key, grn_key, ccc_key)
    ServerInstance.data_hook = datacache
    ServerInstance.front_dir = os.path.dirname(os.path.abspath(__file__)) + '/vt3d_browser'
    print(f'Current front-dir is {ServerInstance.front_dir}', flush=True)
    # create webserver
    server_address = ('', port)
    httpd = StoppableHTTPServer(server_address, DynamicRequstHander)
    ServerInstance.server = httpd
    # start endless waiting now...
    start_url = os.environ.get('stereopy_3d_visualization', None)
    if start_url is None:
        print(f'Starting server on http://{ip}:{port}')
        print(f'To ternimate this server , click: http://{ip}:{port}/endnow')
    else:
        print(f'Starting server on {start_url}')
        print(f'To ternimate this server , click: {start_url}/endnow')
    httpd.serve_forever()
