import base64
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from .constants import (
    BOXPLT_C_INDEX,
    COLORPLT_C_INDEX,
    CMIXT_C_INDEX,
    CT_COLORPLT_INDEX
)
from .utils import timeit


def get_base64_encoded_img(path):
    """
    Get the base64-encoded representation of an image.

    Parameters:
    - path (str): The path of the image file.

    Returns:
    - str: A base64-encoded string representing the image.
    """
    data_uri = base64.b64encode(open(path, 'rb').read()).decode('utf-8')
    return '"data:image/png;base64,{0}"'.format(data_uri)


def get_css():
    """
    Get the CSS styles as a string.

    Returns:
    - str: The CSS styles as a string.
    """

    return r'''
        img {
            width:100%;
            height:100%
        }
        table, th, td {
            border: 0px solid black;
        }
        .bordered {
            border: 1px solid black;
        }
        hr {
            width: 100%;
        }
        .rightpad {
            padding: 3px 15px;
        }
        .shrink > img {
            max-width: 50vw;
        }
        table.center {
            margin-left: auto;
            margin-right: auto;
        }
        footer {
            position: absolute;
            bottom: 0;
            width: 100%;
            height: 2.5rem;
            background-color:#e1c8eb;
            text-align: center;
            padding-bottom:20px
        }
        .content-wrap {
            padding-bottom: 2.5rem;
        }
        .page-container {
            position: relative;
            min-height: 100vh;
        }
        .pad20{
            padding: 0px 20px 0px;
        }
        .centered{
            width: 60%;
            margin: auto;
            text-align: center;
        }
        .centeredSmall{
            width: 70%;
            margin: auto;
            text-align: center;
        }
        .centeredCtPlot{
            width: 50%;
            margin: auto;
            text-align: center;
        }
        .myButton {
            margin-top: 10px;
            margin-bottom:30px;
            box-shadow:inset 0px 1px 0px 0px #efdcfb;
            background-color:#5F0085;
            border-radius:6px;
            border:1px solid #c584f3;
            display:inline-block;
            cursor:pointer;
            color:#ffffff;
            font-family:Arial;
            font-size:15px;
            font-weight:bold;
            padding:6px 24px;
            text-decoration:none;
            text-shadow:0px 1px 0px #9752cc;
        }
        .myButton:hover {
            background:linear-gradient(to bottom, #bc80ea 5%, #dfbdfa 100%);
            background-color:#bc80ea;
        }
        .myButton:active {
            position:relative;
            top:1px;
        }
    '''


def all_slices_get_figure(project_root, figure_name, plotting_level):
    """
    Get the figure as a base64-encoded image.

    This function retrieves the figure located at the specified path within the project root.
    If the plotting level is greater than 0, the function returns the figure as a base64-encoded image string.

    Parameters:
    - project_root (str): The root directory of the project.
    - figure_name (str): The name of the figure file.
    - plotting_level (int): The level of plotting.

    Returns:
    - str: A base64-encoded image string if the figure file exists and the plotting level is greater than 0,
    otherwise an empty string.
    """
    if plotting_level > 0:
        return get_base64_encoded_img(f"{project_root}/{figure_name}") if os.path.isfile(
            f"{project_root}/{figure_name}") else ""


def per_slice_content(path, plotting_level):
    """
    Generate per-slice content.

    This function generates per-slice content by traversing through the directory structure
    rooted at the specified path. It collects table plots from each directory and appends
    them to the content string. The content string is then returned.

    Parameters:
    - path (str): The root path of the directory structure.
    - plotting_level (int): The level of plotting.

    Returns:
    - str: The per-slice content string.
    """

    content = ""
    for root, dirs, files in os.walk(path):
        for name in dirs:
            content += get_table_plots(os.path.join(root, name), plotting_level)
    return content


def get_table_plots(path, plotting_level):
    """
    Get table plots for a given path.

    This function searches for specific files in the provided path and generates an HTML table
    containing the table plots. The table includes celltype_table, cell_mixture_table, and
    window_cell_num_hist plots if the corresponding files are found in the path.

    Parameters:
    - path (str): The path to search for table plot files.

    Returns:
    - str: The HTML table containing the table plots.
    """

    celltype_table = ""
    cell_mixtures = ""
    hist_cell_number = ""
    ct_colorplots = defaultdict(list)
    for root, _, files in os.walk(path):
        for name in files:
            if name.startswith("celltype_table"):
                celltype_table = os.path.join(root, name)
            if name.startswith("cell_mixture_table"):
                cell_mixtures = os.path.join(root, name)
            if name.startswith("window_cell_num_hist"):
                hist_cell_number = os.path.join(root, name)
            if name.startswith("ct_colorplot"):
                ct_colorplots[name.split("_")[CT_COLORPLT_INDEX]].append(os.path.join(root, name))

    return f'''
        <table>
            <thead>
                <tr>
                    <th colspan="3"><h3>slice: {os.path.basename(path)}</h3></th>
                </tr>
            </thead>
            <tbody>
            <tr>
                <td class="pad20" style="width:40%"><a target="_blank" href="#" onClick='open_new_tab(this)'><img src={get_base64_encoded_img(celltype_table) if celltype_table != "" else ""}></a></td>
                <td class="pad20"><a target="_blank" href="#" onClick='open_new_tab(this)'><img src={get_base64_encoded_img(cell_mixtures) if cell_mixtures != "" else ""}></a></td>
                <td class="pad20" style="width:20%"><a target="_blank" href="#" onClick='open_new_tab(this)'></a><img title="Histogram of the number of cells in windows" src={get_base64_encoded_img(hist_cell_number) if hist_cell_number != "" else ""}></a></td>
            </tr>
            </tbody>
        </table>
        <div class="centeredCtPlot">
            {make_table(ct_colorplots, columns=2, comment="Cell type colorplots") if plotting_level > 4 else ""}
        </div>
        <hr>
    '''  # noqa


def per_community_content(path, plotting_level):
    """
    Generate per-community content.

    This function generates per-community content by traversing through the directory
    structure rooted at the specified path. It collects boxplots, cmixtures, and colorplot
    files for each community and organizes them into dictionaries. The content is then
    generated by creating tables for each type of plot. The resulting content string is
    returned.

    Parameters:
    - path (str): The root path of the directory structure.
    - plotting_level (int): The level of plotting.

    Returns:
    - str: The per-community content string.
    """

    if plotting_level < 3:
        return ""
    content = ""
    cmixtures_dict = defaultdict(list)
    boxplots_dict = defaultdict(list)
    colorplot_dict = defaultdict(list)

    for root, dirs, files in os.walk(path):
        for name in dirs:
            slice_path = os.path.join(path, name)
            for file in os.listdir(slice_path):
                if file.startswith("boxplot"):
                    cluster = int(Path(file).stem.split("_")[BOXPLT_C_INDEX][1:])
                    boxplots_dict[cluster].append(os.path.join(slice_path, file))
                if file.startswith("cmixtures"):
                    cluster = int(Path(file).stem.split("_")[CMIXT_C_INDEX][1:])
                    cmixtures_dict[cluster].append(os.path.join(slice_path, file))
                if file.startswith("colorplot"):
                    cluster = int(Path(file).stem.split("_")[COLORPLT_C_INDEX][1:])
                    colorplot_dict[cluster].append(os.path.join(slice_path, file))

    content += make_table(cmixtures_dict, columns=2, comment="Cell types that are present in each community")
    content += make_table(boxplots_dict, columns=2, comment="Boxplots of cell types that are present in each community")
    if plotting_level == 5:
        content += make_table(colorplot_dict, columns=2, comment="RGB Colorplots")
    return content


def make_table(plot_dict, columns, comment):
    """
    Generate an HTML table from a plot dictionary.

    This function generates an HTML table from the provided plot dictionary. Each entry in the
    dictionary represents a set of plots for a specific category. The plots are organized into
    rows and columns based on the specified number of columns. The resulting HTML table is
    returned as a string.

    Parameters:
    - plot_dict (dict): A dictionary containing the plots categorized by keys.
    - columns (int): The number of columns in the table.
    - comment (str): A comment describing the table content.

    Returns:
    - str: The generated HTML table string.
    """

    content = ""
    for _, plots in sorted(plot_dict.items()):
        rows = ""
        plots = [f'<td class="shrink"><img src={get_base64_encoded_img(plot)}></td>' for plot in plots]
        for i in range(0, len(plots), columns):
            row = "".join(plots[i:i + columns])
            rows += f'<tr>{row}</tr>'
        content += f'''
        <table style="border: 1px solid black;">
            <tbody>
            {rows}
            </tbody>
        </table>
        <br>
        '''

    return f'''
        <table>
            <thead>
                <tr>
                    <td colspan="{columns}"><h4>{comment}</h4></td>
                </tr>
            </thead>
        </table>
        {content}
        <hr>
        <br>
    '''


def annotation_and_communities_figures(path):
    """
    Get annotation and communities figures.

    This function searches for annotation and communities figures in the specified directory
    path. It collects the paths of the annotation and communities figures and returns them as
    a tuple. If there are multiple annotation figures, it returns the paths to the aggregated
    cell type per slice and clustering per slice figures.

    Parameters:
    - path (str): The root path of the directory.

    Returns:
    - tuple: A tuple containing the paths of the annotation and communities figures.
    """
    annotations = []
    communities = []
    for root, dirs, _ in os.walk(path):
        for dir in dirs:
            for file in os.listdir(os.path.join(root, dir)):
                if file.startswith("cell_type_anno"):
                    annotations.append(os.path.join(root, dir, file))
                if file.startswith("clusters_cellspots"):
                    communities.append(os.path.join(root, dir, file))
    if len(annotations) > 1:
        return (f"{path}/cell_type_per_slice.png", f"{path}/clustering_per_slice.png")
    else:
        return (annotations[0], communities[0])


@timeit
def generate_report(params):
    """
    Generate a cell communities clustering report.

    This function generates an HTML report for cell communities clustering based on the given parameters.
    The report includes various figures and tables visualizing the clustering results and related data.

    Parameters:
    - params (dict): A dictionary containing the parameters for generating the report.
    """

    if params['plotting'] == 0:
        return

    command = "python main.py "
    for k, v in params.items():
        if v == None or k == 'out_path' or k == 'project_name' or k == "skip_stats" or k == "save_adata":  # noqa
            continue
        if k == 'dev' and v is False:
            continue
        if k == 'out_path_orig' or k == 'project_name_orig':
            k = k[:-5]
        command += f"--{k} {v} "

    dev = params.get('dev', False)

    if dev:
        commit_date = subprocess.check_output(r'git log --pretty=format:"%h%x09%x09%ad%x09%s" -n 1',
                                              shell=True).expandtabs()
        created_time = f"Report created from commit: {commit_date}"
    else:
        commit_date = time.strftime("%I:%M%p  %B %d, %Y")
        created_time = f"Report created at {commit_date}"

    annotation_figure, communities_figure = annotation_and_communities_figures(params['out_path'])

    htmlstr = f'''
    <!DOCTYPE html>

    <html lang=”en”>
    <head>
    <style>
    {get_css()}
    </style>

    <script>
    let text = "{command}"

    const copyCommand = async () => {{
        try {{
        await navigator.clipboard.writeText(text);
        console.log('Command copied to clipboard');
        }} catch (err) {{
        console.error('Failed to copy: ', err);
        }}
    }}

    function open_new_tab(element) {{
        var newTab = window.open();
        setTimeout(function() {{
            newTab.document.body.innerHTML = element.innerHTML;
        }}, 500);
        return false;
    }}

    function remove_elements(){{
    let allEls = document.querySelectorAll('.testRemove')
    allEls.forEach(el => {{
        if (el.getAttribute('data-value') == "remove"){{
            el.style.display = 'none';
            }}
        }});
        }}
    </script>

    <title>Cell Communities Report</title>
    <link rel="icon" type="image/png" sizes="16x16" href={get_base64_encoded_img(f"{os.path.dirname(__file__)}/assets/favicon.ico")}/>
    </head>

    <body onload="remove_elements()">
    <div class="page-container">
        <div class="content-wrap">
            <div style="background-color:#e1c8eb;">
                <h1 style="text-align: center; margin:0"> Cell communities clustering report</h1>
                <br>
                <table class="center">
                    <thead>
                        <tr>
                            <th colspan="2"><h3>Parameters used</h3></th>
                        </tr>
                        <tr>
                            <th colspan="2"><hr></th>
                        </tr>
                    </thead>
                    <tbody>
                    <tr>
                        <td class="rightpad">Annotation</td>
                        <td>{params['annotation']}</td>
                    </tr>
                    <tr>
                        <td class="rightpad">Resolution</td>
                        <td>{params['resolution']}</td>
                    </tr>
                    <tr>
                        <td class="rightpad">Spot size</td>
                        <td>{params['spot_size']}</td>
                    </tr>
                    <tr>
                        <td class="rightpad">Total cells normalization</td>
                        <td>{params['total_cell_norm']}</td>
                    </tr>
                    <tr>
                        <td class="rightpad">Downsample rate</td>
                        <td>{params['downsample_rate']}</td>
                    </tr>
                    <tr>
                        <td class="rightpad">Entropy threshold</td>
                        <td>{params['entropy_thres']}</td>
                    </tr>
                    <tr>
                        <td class="rightpad">Scatteredness threshold</td>
                        <td>{params['scatter_thres']}</td>
                    </tr>
                    <tr>
                        <td class="rightpad">Window sizes</td>
                        <td>{params['win_sizes']}</td>
                    </tr>
                    <tr>
                        <td class="rightpad">Sliding steps</td>
                        <td>{params['sliding_steps']}</td>
                    </tr>
                    <tr>
                    <td colspan="2"><button title="Command that is used to generate results is copied to clipboard" class="rightpad myButton" type="button" onclick="copyCommand()">
                    Copy command</button></td>
                    </tr>
                    </tbody>
                </table>
            </div>
            <br><hr>
            <div>
                <h2 style="text-align: center;">Overall</h2>
            </div>
            <div class="centered">
                <h3>Cell type annotation:</h3>
            </div>
            <div class="centered">
                <a target="_blank" href="#" onClick='open_new_tab(this)'>
                <img title="Annotation column in .obs of anndata object" src={get_base64_encoded_img(annotation_figure)}>
                </a>
            </div>
            <div class="centered">
                <h3>Communities obtained:</h3>
            </div>
            <div class="centered">
                <a target="_blank" href="#" onClick='open_new_tab(this)'>
                <img title="Result of CCD algorithm contained in .obs of anndata object" src={get_base64_encoded_img(communities_figure)}>
                </a>
            </div>
            <div class="centered testRemove" data-value={"remove" if params['plotting'] < 3 else "keep"}>
                <h3>Cell mixtures for all slices:</h3>
            </div>
            <div class="centered testRemove" data-value={"remove" if params['plotting'] < 3 else "keep"}>
                <a target="_blank" href="#" onClick='open_new_tab(this)'>
                <img title="Showing the percentage of each cell type within obtained community and corresponding sums" src={all_slices_get_figure(params['out_path'], "total_cell_mixtures_table.png", params['plotting'])}>
                </a>
            </div>
            <div class="centered testRemove" data-value={"remove" if params['plotting'] < 3 else "keep"}>
                <h3>Cell type abundance:</h3>
            </div>
            <div class="centered testRemove" data-value={"remove" if params['plotting'] < 3 else "keep"}>
                <a target="_blank" href="#" onClick='open_new_tab(this)'>
                <img title="Percentage of specific cell types per slice" src={all_slices_get_figure(params['out_path'], "cell_abundance_all_slices.png", params['plotting'])}>
                </a>
            </div>
            <div class="centered testRemove" data-value={"remove" if params['plotting'] < 3 else "keep"}>
                <h3>Communities abundance:</h3>
            </div>
            <div class="centered testRemove" data-value={"remove" if params['plotting'] < 3 else "keep"}>
                <a target="_blank" href="#" onClick='open_new_tab(this)'>
                <img title="Percentage of cells in each community per slice" src={all_slices_get_figure(params['out_path'], "cluster_abundance_all_slices.png", params['plotting'])}>
                </a>
            </div>
            <div class="centered testRemove" data-value={"remove" if params['plotting'] < 4 else "keep"}>
                <h3>Cell percentage in communities:</h3>
            </div class="centered testRemove">
            <div class="centered testRemove" data-value={"remove" if params['plotting'] < 4 else "keep"}>
                <a target="_blank" href="#" onClick='open_new_tab(this)'>
                <img title="Percentage of cells in each community per slice" src={all_slices_get_figure(params['out_path'], "cell_perc_in_community_per_slice.png", params['plotting'])}>
                </a>
            </div>
            <br><hr>
            <div data-value={"remove" if params['plotting'] < 2 else "keep"}>
                <h2 style="text-align: center;">Per slice</h2>
            </div>
            <div class="testRemove" data-value={"remove" if params['plotting'] < 2 else "keep"}>
                {per_slice_content(params['out_path'], params['plotting'])}
            </div>
            <br><hr>
            <div class="testRemove" data-value={"remove" if params['plotting'] < 3 else "keep"}>
                <h2 style="text-align: center;">Per community</h2>
            </div>
            <div class="centeredSmall" data-value={"remove" if params['plotting'] < 3 else "keep"}>
                {per_community_content(params['out_path'], params['plotting'])}
            </div>
        </div>
        <footer><h4>{created_time}</h4></footer>
    </div>
    </body>
    </html>
    '''  # noqa

    with open(f"{params['out_path']}/report.html", "w") as f:
        f.write(htmlstr)
