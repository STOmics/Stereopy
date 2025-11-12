from typing import Union, List

import numpy as np
import pandas as pd
import os
import plotly
import plotly.graph_objects as go
import plotly.express as px
import anndata as ad
from typing import List, Optional
from anndata import AnnData
from .utils import pre_check_adata

from stereo.core.stereo_exp_data import StereoExpData

def animate_transfer(
    data_list: List[StereoExpData],
    # transfer_path: str,
    transfer_data: Union[pd.DataFrame, str],
    fig_title: str,  # name of fig, "'RGC' transfer from E12.5 E14.5 to E16.5"
    save_path: str,  # path to save fig, "./Dorsal_result_ot/plotly_dark_1209.html"
    #group:str, ## annotation
    spatial_key:str='spatial',
    color:list=[],
    time:str='time',
    annotation:str='celltype',
    times: Optional[List[str]] = None, # order time(start time -> end time) ['E12.5','E14.5','E16.5']
    N: int = 2,
    n: int = 6,  # number of point to shoe on the transfer line 
):
    # Load transfer data,(columns:slice1,slice1_x,slice1_y,slice1_anno,k_line,b_line,......)
    if isinstance(transfer_data, pd.DataFrame):
        df1 = transfer_data
    elif os.path.isfile(transfer_data):
        df1 = pd.read_csv(transfer_data, index_col=0)
    else:
        raise ValueError("transfer_data should be a path of csv file or a pandas DataFrame.")
    # load adata_list from generate_animate_input.
    for i, data in enumerate(data_list):
        
        if data.cells.obs.columns.isin(['x','y','time','cell_id','annotation']).sum() == 5:
            pass
        else:
            data_list[i]= pre_check_adata(data, spatial_key=spatial_key, time=time, annotation=annotation)
    # adatas = ad.concat(adata_list)
    # df = adatas.obs[['x', 'y', 'time', 'annotation']]
    obs_list = [i.cells.obs[['x', 'y', 'time', 'annotation']] for i in data_list]
    df = pd.concat(obs_list)
    slice_list = df1.columns.tolist()[:len(data_list)]
    

    # Generate line discrete data on transfer trace.
    x = {}
    y = {}
    xx = {}
    yy = {}
    trace_line = []

    for i in range(df1.shape[0]):
        x[i] = []
        y[i] = []
        xx[i] = []
        yy[i] = []
        for j, element in enumerate(slice_list):
            if j >=1 and j <= len(slice_list)-1: 
                x[i] = np.append(x[i],np.linspace(df1["slice"+str(j)+"_x"][i], df1["slice"+str(j+1)+"_x"][i], N))
                y[i] = np.append(y[i],df1['k_line_'+str(j)+str(j+1)][i]*x[i][(j-1)*N:j*N] + df1['b_line_'+str(j)+str(j+1)][i])
                xx[i] = np.append(xx[i],np.linspace(df1["slice"+str(j)+"_x"][i], df1["slice"+str(j+1)+"_x"][i], n))
                yy[i] = np.append(yy[i],df1['k_line_'+str(j)+str(j+1)][i]*xx[i][(j-1)*n:j*n] + df1['b_line_'+str(j)+str(j+1)][i])
        trace_line.append(go.Scatter(x=x[i], y=y[i], mode="lines", showlegend=False, line=dict(width=0.000001, color="blue")))

    xm = df1[[i for i in df1.columns if i.endswith("x")]].min().min()
    xM = df1[[i for i in df1.columns if i.endswith("x")]].max().max()
    ym = df1[[i for i in df1.columns if i.endswith("y")]].min().min()
    yM = df1[[i for i in df1.columns if i.endswith("y")]].min().min()

    # Load the time data
    if times is None:
        times = df['time'].unique()
    num_times = len(times)
    if len(color)>0:
        pass
    else:
        color = ['#99cfdd', '#fe3030', '#04047d', '#95c454', '#666666', '#ccc8fb', '#1c9d79', '#775613', '#b58396', '#02ffff', '#AF5F3C', '#e9c62a', '#525510', '#4fbad6',
             '#D1D1D1', '#FEB915', '#6D1A9C', '#fffe2e', '#4ec602', '#b904ab', '#4166b0', '#59BE86', '#ffd29a', '#C798EE', '#e72988', '#8eb3fb', '#767da3', '#828282']
    color_group_dict = dict(zip(set(df['annotation'].tolist()), color))
    
    # the start time data
    trace_time0 = []
    for i in set(df['annotation'].tolist()):
        tips_temp = df.loc[df['annotation'] == i, :]  # annotation color
        trace_time0.append(go.Scatter(x=tips_temp[tips_temp['time'].isin([times[0]])]['x'],
                                      y=tips_temp[tips_temp['time'].isin(
                                          [times[0]])]['y'],
                                      mode='markers',
                                      name=i,
                                      marker=dict(color=color_group_dict[i],
                                                  size=1,
                                                  showscale=False,),
                                      ))
        
    trace_times = []
    for t in range(len(times)-1):
        globals()['trace_'+str(t+1)] = go.Scatter(x=df[df['time'].isin([times[t+1]])].drop(labels=df1['slice'+str(t+2)].tolist(), axis=0)['x'],
                                                  y=df[df['time'].isin([times[t+1]])].drop(labels=df1['slice'+str(t+2)].tolist(), axis=0)['y'],
                                                  mode='markers',
                                                  marker=dict(size=1,
                                                              color='grey',
                                                              showscale=False,
                                                              ),
                                                  showlegend=False
                                                  )
        globals()['trace_'+str(t+1)+'_add'] = []
        for i in range(df1.shape[0]):
            globals()['trace_'+str(t+1)+'_add'].append(go.Scatter(x=[xx[i][(t+1)*n-1]],
                                                                  y=[yy[i][(t+1)*n-1]],
                                                                  mode="markers",
                                                                  marker=dict(size=1,
                                                                              color=[color_group_dict[df1['slice'+str(t+2)+'_annotation'][i]]]
                                                                             ),
                                                                  showlegend=False
                                                                 )
                                                      )
        globals()['trace_'+str(t+1)] = [globals()['trace_'+str(t+1)]] + globals()['trace_'+str(t+1)+'_add']
        trace_times = trace_times + globals()['trace_'+str(t+1)]
    
    # Generate animate figure
    # Frames
    fram = []
    for j in range(num_times-1):
        fram = fram+[go.Frame(data=[go.Scatter(x=[xx[i][k]],
                                               y=[yy[i][k]],
                                               mode="markers",
                                               marker=dict(color=[color_group_dict[df1['slice'+str(j+1)+'_annotation'][i]]], size=1),) for i in range(df1.shape[0])]) 
                    for k in range(j*n, (j+1)*n)]
    # animate figure
    fig = go.Figure(
        data=trace_line + trace_time0 + trace_times,
        layout=go.Layout(
            xaxis=dict(range=[xm, xM], autorange=True, zeroline=False),
            yaxis=dict(range=[ym, yM], autorange=True, zeroline=False),
            title_text=fig_title, hovermode="closest",
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [None, {"frame": {"duration": 500, "redraw": False},
                                            "fromcurrent": True, "transition": {"duration": 300,
                                                                                "easing": "quadratic-in-out"}}],
                            "label": "Play",
                            "method": "animate"
                        },
                        {
                            "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                              "mode": "immediate",
                                              "transition": {"duration": 0}}],
                            "label": "Pause",
                            "method": "animate"
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }
            ]),
        frames=fram
    )
    fig.update_layout(autosize=False,
                      width=800,
                      height=500,
                      legend=dict(
                          title_font_family="Arial",
                          font=dict(
                              family="Arial",
                              size=10,
                          ),
                          itemsizing='constant'
                      ),
                      # template="simple_white",
                      template="plotly_dark",
                      # template="none",
                      )
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, showticklabels=False)
    # save fig
    plotly.offline.plot(fig,
                        filename=save_path,
                        auto_open=False,
                        )
    return fig
