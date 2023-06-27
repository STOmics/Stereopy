import atexit
import os
import sys

from ..plot_base import PlotBase


class Plot3DBrowser(PlotBase):

    def start_vt3d_browser(self, port=7654, cluster_label='annotation', spatial_label='spatial_rigid', mesh_method='delaunay_3d'):
        pid = _daemonize()
        if not pid:
            from .stereopy_3D_browser import launch
            launch(self.stereo_exp_data, meshes=self.stereo_exp_data.tl.result['mesh'][mesh_method],
                   cluster_label=cluster_label, spatial_label=spatial_label, port=port)

    def display_3d_mesh(self, width=1400, height=1200, port=7654):
        import IPython
        return IPython.display.IFrame(src=f"http://127.0.0.1:{port}", width=width, height=height)


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
