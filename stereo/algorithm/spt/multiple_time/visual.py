import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from random import sample
import seaborn as sns

'''
function: 3D visualization using three slices

'''

def visual_3D_mapping_3(df_concat,df_mapping_res,
                        point_size=0.8,line_width=0.03,line_alpha=0.8,
                        view_axis=None,
                        cell_color_list= list(map(mpl.colors.rgb2hex, sns.color_palette('tab20b', 20)))):

    """
    Parameters
    ----------
    df_concat: pandas dataframe
           Inlcude cellid, cell annotation and cell spatial coordinate of all slices.
    df_mapping_res: pandas dataframe
            Transfered cell id of all slices 

    point_size: float
            Each cell point size in scatter plot
    line_width: float
            Line width between transfered cellsc
    cell_color_list: list
            Cell type color
    view_axis: 'x','y'or'z'
            Set angle of view from 'x','y' or 'z'
    """
    df_mapping_res=df_mapping_res[['slice1','slice2','slice3']]
    df_mapping_res=df_mapping_res.reset_index(drop=True)
    plt.rcParams["figure.figsize"] = [10, 7]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    cell_type_col_name=df_concat.columns[1]
    for slice_id in df_mapping_res.columns:   

        df1=pd.DataFrame(df_mapping_res[slice_id])
        df1.columns=[df_concat.columns[0]]
        df_merge=pd.merge(df1,df_concat.loc[df_concat['batch']==slice_id])
        df_mapping_res[[slice_id+'_cluster',slice_id+'_x',slice_id+'_y']]=df_merge[[cell_type_col_name,'x','y']]

    df_mapping_res=df_mapping_res.dropna()
    if len(df_mapping_res)>400:
        df_mapping_res=df_mapping_res.loc[sample(list(df_mapping_res.index),400)]
    else:
        pass
    color_cell = dict(zip(list(sorted(list(set(df_concat[cell_type_col_name])))),cell_color_list))

    distance_list=list()
    distance=0
    for batch in df_mapping_res.columns[0:3].sort_values():
        df_adata=df_concat[df_concat['batch']==batch]
        distance=distance+0.5
        distance_list.append(distance)
        for i in range(len(df_adata)):
            if df_adata[df_concat.columns[0]].iloc[i] in list(df_mapping_res[batch]):
                pass
            else:
                ax.scatter(distance,df_adata['x'].iloc[i], df_adata['y'].iloc[i], edgecolors=color_cell[df_adata['annotation'].iloc[i]],s=point_size,facecolors='none',linewidth=point_size/3)

    for i in range(len(df_mapping_res)):
        x_values = [df_mapping_res['slice1_x'].iloc[i], df_mapping_res['slice2_x'].iloc[i],df_mapping_res['slice3_x'].iloc[i]]
        y_values = [df_mapping_res['slice1_y'].iloc[i],df_mapping_res['slice2_y'].iloc[i],df_mapping_res['slice3_y'].iloc[i]]
        x,y,z= distance_list,x_values, y_values
        ax.scatter(x[0], y[0],z[0], c=color_cell[df_mapping_res['slice1_cluster'].iloc[i]], s=point_size/4,marker='^')
        ax.scatter(x[1], y[1],z[1],c=color_cell[df_mapping_res['slice2_cluster'].iloc[i]], s=point_size/4,marker='^')
        ax.scatter(x[2], y[2],z[2],c=color_cell[df_mapping_res['slice3_cluster'].iloc[i]], s=point_size/4,marker='^')
        ax.plot([x[0],x[1]], [y[0],y[1]],[z[0],z[1]], linewidth=line_width,color=color_cell[df_mapping_res['slice1_cluster'].iloc[i]],alpha=line_alpha)
        ax.plot([x[1],x[2]], [y[1],y[2]],[z[1],z[2]], linewidth=line_width,color=color_cell[df_mapping_res['slice2_cluster'].iloc[i]],alpha=line_alpha)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('x',fontsize=25)
    ax.set_ylabel('y',fontsize=25)
    ax.set_zlabel('z',fontsize=25)
    if view_axis==None:
        pass
    else:
        ax.view_init(vertical_axis=view_axis) 
    return fig



#Visual_3D_mapping_3(df_concat,df_mapping,point_size=0.8,line_width=0.03,cell_color_list=['#9A154C', '#F3754E', '#5B58A7'])
#plt.savefig('/hwfssz1/ST_SUPERCELLS/P21Z10200N0134/USER/huangke2/27.spatial.trajectory/12.SpaTrack/07.ICC.new/06.mouse.brain/OT.mapping/3D.mouse.1.pdf')

'''
function: 3D visualization using two slices
'''

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from random import sample
import seaborn as sns


def visual_3D_mapping_2(
    df_concat,
    df_mapping,
    point_size=0.8,
    line_width=0.03,
    line_alpha=0.8, 
    view_axis=None,
    cell_color_list= list(map(mpl.colors.rgb2hex, sns.color_palette('tab20b', 20)))
):
    
    """
    Parameters
    ----------
    df_concat: pandas dataframe
           Inlcude cellid, cellannoand cell spatial coordinate of all slices.
    df_mapping: pandas dataframe
            Transfered cell id of all slices 

    point_size: float
            Each cell point size in scatter plot
    line_width: float
            Line width between transfered cellsc
    cell_color_list: list
            Cell type color
    view_axis: 'x','y'or'z'
            Set angle of view from 'x','y' or 'z'
    """
 
    plt.rcParams["figure.figsize"] = [10, 7]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    df_mapping.columns=['slice1','slice2']
    df_mapping=df_mapping.reset_index(drop=True)
    cell_type_col_name=df_concat.columns[1]
    for slice_id in df_mapping.columns:   

        df1=pd.DataFrame(df_mapping[slice_id])
        df1.columns=[df_concat.columns[0]]
        df_merge=pd.merge(df1,df_concat.loc[df_concat['batch']==slice_id])
        df_mapping[[slice_id+'_cluster',slice_id+'_x',slice_id+'_y']]=df_merge[[cell_type_col_name,'x','y']]

    df_mapping=df_mapping.dropna()
    if len(df_mapping)>300:
        df_mapping=df_mapping.loc[sample(list(df_mapping.index),300)]
    else:
        pass
    color_cell = dict(zip(list(sorted(list(set(df_concat[cell_type_col_name])))),cell_color_list))

    distance_list=list()
    distance=0
    for batch in df_mapping.columns[0:2].sort_values():
        df_adata=df_concat[df_concat['batch']==batch]
        distance=distance+0.5
        distance_list.append(distance)
        for i in range(len(df_adata)):
            if df_adata[df_concat.columns[0]].iloc[i] in list(df_mapping[batch]):
                pass
            else:
                ax.scatter(distance,df_adata['x'].iloc[i], df_adata['y'].iloc[i], edgecolors=color_cell[df_adata[cell_type_col_name].iloc[i]],s=point_size,facecolors='none',linewidth=point_size/3)


    for i in range(len(df_mapping)):
        x_values = [df_mapping['slice1_x'].iloc[i], df_mapping['slice2_x'].iloc[i]]
        y_values = [df_mapping['slice1_y'].iloc[i],df_mapping['slice2_y'].iloc[i]]
        x,y,z= distance_list,x_values, y_values
        ax.scatter(x[0], y[0],z[0], c=color_cell[df_mapping['slice1_cluster'].iloc[i]], s=point_size/4,marker='^')
        ax.scatter(x[1], y[1],z[1],c=color_cell[df_mapping['slice2_cluster'].iloc[i]], s=point_size/4,marker='^')
        ax.plot(x, y, z, linewidth=line_width,color=color_cell[df_mapping['slice1_cluster'].iloc[i]],alpha=line_alpha)


    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('x',fontsize=25)
    ax.set_ylabel('y',fontsize=25)
    ax.set_zlabel('z',fontsize=25)

    if view_axis==None:
        pass
    else:
        ax.view_init(vertical_axis=view_axis)
    return fig
