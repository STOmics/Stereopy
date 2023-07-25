import atexit
import os
import sys
from time import sleep
from threading import Thread

from ..plot_base import PlotBase
from .stereopy_3D_browser import launch


class Plot3DBrowser(PlotBase):

    def start_vt3d_browser(
        self,
        port=7654,
        paga_res_key = None,
        ccc_res_key = None,
        grn_res_key = None,
        cluster_res_key = None,
        mesh_method='delaunay_3d'
    ):
        if paga_res_key is not None:
            meshes = self.stereo_exp_data.tl.result['mesh'][mesh_method]
        else:
            meshes = {}
        th = Thread(
            target=launch,
            args=(self.stereo_exp_data, ),
            kwargs={
                'meshes': meshes,
                'cluster_label': cluster_res_key,
                'paga_key': paga_res_key,
                'ccc_key': ccc_res_key,
                'grn_key': grn_res_key,
                'port': port
            }
        )
        th.setDaemon(True)
        th.start()
        # pid = _daemonize()
        # if not pid:
        #     launch(
        #         self.stereo_exp_data,
        #         meshes=meshes,
        #         cluster_label=cluster_res_key,
        #         paga_key=paga_res_key,
        #         ccc_key=ccc_res_key,
        #         grn_key=grn_res_key,
        #         port=port
        #     )
        # launch(
        #     self.stereo_exp_data,
        #     meshes=meshes,
        #     paga_key=paga_res_key,
        #     ccc_key=ccc_res_key,
        #     grn_key=grn_res_key,
        #     port=port
        # )

    def display_3d_mesh(self, width=1400, height=1200, port=7654):
        import IPython
        sleep(5)
        return IPython.display.IFrame(src=f"http://127.0.0.1:{port}", width=width, height=height)
    
    def display_3d_ccc(self, *args, **kwargs):
        return self.display_3d_mesh(*args, **kwargs)
    
    def display_3d_grn(self, *args, **kwargs):
        return self.display_3d_mesh(*args, **kwargs)


def _daemonize():
    pid = os.fork()
    if pid:
        return pid

    sys.stdout.flush()
    sys.stderr.flush()

    with open('/dev/null') as read_null, open('/dev/null', 'w') as write_null:
        os.dup2(read_null.fileno(), sys.stdin.fileno())
        os.dup2(write_null.fileno(), sys.stdout.fileno())
        os.dup2(write_null.fileno(), sys.stderr.fileno())

    pid_file = "vt3d_browser.pid"
    if pid_file:
        with open(pid_file, 'w+') as f:
            f.write(str(os.getpid()))
        atexit.register(os.remove, pid_file)
