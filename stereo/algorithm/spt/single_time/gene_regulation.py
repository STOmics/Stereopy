import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import random
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import sys
from scipy.sparse import issparse
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

from typing import Literal, Union, List

from stereo.core.stereo_exp_data import StereoExpData
from stereo.preprocess.filter import filter_cells, filter_genes
from stereo.log_manager import logger


class Model(nn.Module):
    """
    Model for exploring the relationship between TFs and genes.

    Parameters
    ----------
    n_gene
        The dimensionality of the input, i.e. the number of genes.
    n_TF
        The dimensonality of the output, i.e. the number of TFs.
    """

    def __init__(
        self,
        n_gene: int,
        n_TF: int,
    ) -> None:
        super(Model, self).__init__()
        self.n_gene = n_gene
        self.n_TF = n_TF

        self.linear = nn.Linear(n_gene, n_TF)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Give the gene changes of cells, and get the relationship between genes and TFs through linear regression.

        Parameters
        ----------
        x
            The input data (gene changes)

        Returns
        -------
        :class:`torch.Tensor`

                    Tensors for the output data (TF expresssion):
        """
        y_pred = self.linear(x)
        return y_pred


class Trainer:
    """
    Class for implementing the training process.

    parameters
    ----------
    type
        The Data type. Including dual time point data of two slices and pseudotime data of one slice.
    expression_matrix_path
        The path of the expression matrix file.
    tfs_path
        The path of the tf names file.
    cell_mapping_path
        The path of the cell mapping file, where column `slice1` indicates the start cell and column `slice2` indicates the end cell.
    ptime_path
        The path of the ptime file, used to determine the sequence of the ptime data.
    min_cells, optional
        The minimum number of cells for gene filtration.
    cell_divide_per_time, optional
        The cell number generated at each time point using the meta-analysis method, by default 500.
    cell_select_per_time, optional
        The number of randomly selected cells at each time point.
    cell_generate_per_time, optional
        The number of cells generated at each time point.
    train_ratio
        Ratio of training data.
    use_gpu, optional
        Whether to use gpu, by default True.
    random_state
        Random seed of numpy and torch.
    """

    def __init__(
        self,
        data_type: Literal["2_time", "p_time"],
        data: Union[StereoExpData, List[StereoExpData]],
        tfs_path: str = None,
        cell_mapping: pd.DataFrame = None,
        ptime_path: str = None,
        min_cells: Union[int, List[int]] = None,
        cell_divide_per_time: int = 80,
        cell_select_per_time: int = 10,
        cell_generate_per_time: int = 500,
        train_ratio: float = 0.8,
        use_gpu: bool = True,
        random_state: int = 0,
    ) -> None:
        self.train_ratio = train_ratio

        gpu = torch.cuda.is_available() and use_gpu
        if gpu:
            torch.cuda.manual_seed(random_state)
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        np.random.seed(random_state)
        random.seed(random_state)
        torch.manual_seed(random_state)

        # read gene expression, cell ptime and tfs files.
        if data_type == "p_time":
            # if type(expression_matrix_path) == list:
            #     expression_matrix_path = expression_matrix_path[0]
            # self.adata = sc.read(expression_matrix_path)
            # self.adata.obs["ptime"] = pd.read_table(ptime_path, index_col=0)
            # self.tfs_path = tfs_path

            if type(data) == list:
                data = data[0]
            self.data = data
            self.data.cells['ptime'] = pd.read_csv(ptime_path, index_col=0)
            self.tfs_path = tfs_path

            if min_cells == None:
                # min_cells = round(self.adata.n_obs*0.3)
                min_cells = round(self.data.n_cells * 0.3)

            # filter out genes expressed in few cells
            # sc.pp.filter_genes(self.adata, min_cells=min_cells, inplace=True)
            filter_genes(self.data, min_cells=min_cells, inplace=True)
            logger.info(f"Genes expressed in less than {min_cells} cells have been filtered.")

            all_tfs = pd.read_csv(self.tfs_path,header=None)
            # self.genes = self.adata.var_names
            self.genes = self.data.genes.var.index
            self.tfs = self.genes.intersection(all_tfs[0].tolist())

            self.generate_data_p_time(
                cell_divide_per_time, cell_select_per_time, cell_generate_per_time
            )
        elif data_type == "2_time":
            # 2_time Pattern need two adata files, the cell mapping file and the tfs file.
            if type(data) != list or len(data) != 2:
                raise Exception(
                    "In 2_time mode, you should provide a list contains two data objects through the data parameter."
                )

            # self.adata1 = sc.read(expression_matrix_path[0])
            # self.adata2 = sc.read(expression_matrix_path[1])
            # self.cell_mapping = pd.read_csv(cell_mapping_path, index_col=0)
            self.data1 = data[0]
            self.data2 = data[1]
            self.cell_mapping = cell_mapping
            self.tfs_path = tfs_path

            # filter genes
            if min_cells == None:
                min_cells = [round(self.data1.n_cells*0.3), round(self.data2.n_cells*0.3)]
            if type(min_cells) != list or len(min_cells) != 2:
                raise Exception(
                    "In 2_time mode, you should provide a list contains two min_cells parameters to filter two data objects."
                )

            # sc.pp.filter_genes(self.adata1, min_cells=min_cells[0], inplace=True)
            # sc.pp.filter_genes(self.adata2, min_cells=min_cells[1], inplace=True)
            filter_genes(self.data1, min_cells=min_cells[0], inplace=True)
            filter_genes(self.data2, min_cells=min_cells[1], inplace=True)

            # only use the mapping cells
            # self.adata1 = self.adata1[self.cell_mapping.slice1]
            # self.adata2 = self.adata2[self.cell_mapping.slice2]
            # filter_cells(self.data1, cell_list=self.cell_mapping['slice1'], inplace=True)
            # filter_cells(self.data2, cell_list=self.cell_mapping['slice2'], inplace=True)

            # get same genes
            # same_genes = list(self.adata1.var_names & self.adata2.var_names)
            # self.adata1 = self.adata1[:, same_genes]
            # self.adata2 = self.adata2[:, same_genes]
            # self.genes = self.adata1.var_names
            same_genes = list(self.data1.genes.var.index & self.data2.genes.var.index)
            # filter_genes(self.data1, gene_list=same_genes, inplace=True)
            # filter_genes(self.data2, gene_list=same_genes, inplace=True)
            self.data1 = self.data1.sub_by_name(cell_name=self.cell_mapping['slice1'], gene_name=same_genes, copy=False)
            self.data2 = self.data2.sub_by_name(cell_name=self.cell_mapping['slice2'], gene_name=same_genes, copy=False)
            self.genes = self.data1.genes.var.index

            # get tfs
            all_tfs = pd.read_table(self.tfs_path, header=None)
            self.tfs = self.genes.intersection(all_tfs[0].tolist())

            self.generate_data_2_time(cell_generate_per_time, cell_select_per_time)

    def getMetaData(self, cell_generate_per_time, cell_select_per_time) -> None:
        """
        Randomly sample cell expression and generate meta data.

        Parameters
        ----------
        cell_generate_per_time
            The amount of data generated at each time point.
        cell_select_per_time
            The amount of data randomly selected at each time point.
        """
        new_data_in = []
        new_data_out = []
        for i in range(cell_generate_per_time):
            cell_indexes = random.sample(
                range(self.input_data.shape[0]), cell_select_per_time
            )
            new_data_in.append(np.mean(self.input_data[cell_indexes], axis=0))
            new_data_out.append(np.mean(self.output_data[cell_indexes], axis=0))

        self.input_data = np.stack(new_data_in)
        self.output_data = np.stack(new_data_out)

    def run(
        self,
        training_times: int = 10,
        iter_times: int = 30,
        mapping_num: int = 3000,
        filename: str = "weights.csv",
        lr_ratio: float = 0.1
    ) -> None:
        """
        Run the trainer.

        Parameters
        ----------
        training_times
            Number of times to randomly initialize the model and retrain.
            (Default: 10)
        iter_times
            The number of iterations for each training model, by default 30.
            (Default: 30)
        mapping_num
            The number of top weight pairs you want to extract.
            (Default: 3000)
        filename
            The saved file name.
            (Default: 'weights.csv')
        """
        self.mapping_num = mapping_num
        self.all_gtf = []

        with tqdm(total=training_times * iter_times) as pbar:
            for i in range(training_times):
                pbar.set_description(f"Train {i + 1}")
                self.model = Model(len(self.genes), len(self.tfs)).to(self.device)
                loss_fn = nn.MSELoss()
                optimizer = torch.optim.SGD(self.model.parameters(), lr=lr_ratio)

                for t in range(iter_times):
                    train_loss = self.train(
                        self.train_dl, self.model, loss_fn, optimizer
                    )
                    test_loss = self.test(self.test_dl, self.model, loss_fn)

                    pbar.set_postfix(
                        {"train_loss": train_loss, "test_loss": test_loss}, refresh=True
                    )
                    pbar.update()
                gtf = self.model.linear.weight.T
                self.all_gtf.append(gtf)

        # set the highest weighted map (TF itself) to 0
        sum_all_gtf = torch.mean(torch.stack(self.all_gtf), dim=0)
        _, self_idx = torch.max(sum_all_gtf, dim=0)
        for i in range(len(self_idx)):
            sum_all_gtf[self_idx[i], i] = 0

        # save most important maps
        flat_tensor = torch.flatten(sum_all_gtf)
        sorted_tensor, _ = torch.sort(flat_tensor, descending=True)
        if len(sorted_tensor) <= mapping_num:
            mapping_num = len(sorted_tensor) - 1
            print(f"Only got {mapping_num+1} pairs of weights.")

        max_value = sorted_tensor[mapping_num]
        # min_value = sorted_tensor[-(mapping_num + 1)]

        self.max_TF_idx = torch.nonzero(sum_all_gtf > max_value)
        # self.min_TF_idx = torch.nonzero(sum_all_gtf < min_value)

        network_rows = []
        for i in self.max_TF_idx:
            gene = self.genes[i[0].item()]
            TF = self.tfs[i[1].item()]
            weight = sum_all_gtf[i[0].item(), i[1].item()].item()
            one_row = [TF, gene, weight]
            network_rows.append(one_row)
        # for i in self.min_TF_idx:
        #     gene = self.genes[i[0].item()]
        #     TF = self.tfs[i[1].item()]
        #     weight = sum_all_gtf[i[0].item(), i[1].item()].item()
        #     one_row = [TF, gene, weight]
        #     network_rows.append(one_row)

        columns = ["TF", "gene", "weight"]
        self.network_df = pd.DataFrame(data=network_rows, columns=columns)
        self.network_df.sort_values(by="weight", ascending=False, inplace=True)
        self.network_df.to_csv(filename, index=0)
        logger.info(f"Weight relationships of tfs and genes are stored in {filename}.")


    def lm(self, gene_name, TF_name, ax: Axes):
        gene_loc = self.genes.get_loc(gene_name)
        TF_loc = self.tfs.get_loc(TF_name)

        train_x = self.output_data[:, TF_loc]   # TF
        train_y = self.input_data[:, gene_loc]  # gene

        # zero_TF_idx=np.where(train_x==0)[0]

        # train_x = np.delete(train_x,zero_TF_idx)
        # train_y = np.delete(train_y, zero_TF_idx)
        # print(train_x.shape,train_y.shape)

        theta0=np.random.rand()
        theta1=np.random.rand()

        def f(x):
            return theta0+theta1*x

        def E(x,y):
            return 0.5*np.sum((y-f(x))**2)
        
        ETA=1e-4
        diff=1
        count=0
        error=E(train_x,train_y)
        while diff>1E-2:
            tmp0=theta0-ETA*np.sum((f(train_x)-train_y))
            tmp1=theta1-ETA*np.sum((f(train_x)-train_y)*train_x)
            theta0=tmp0
            theta1=tmp1
            current_error=E(train_x,train_y)
            diff=error-current_error
            error=current_error

            count+=1
        x=np.linspace(0,1,100)

        ax.plot(train_x, train_y, 'o', markersize=3, color='black', alpha=0.5)
        ax.plot(x, f(x), color='#e87d72', linewidth=3)
        ax.set_xlabel(TF_name, fontsize=14)
        ax.set_ylabel('Dynamics of '+gene_name, fontsize=14)
        ax.axis('equal')
        # return plt
    

    def plot_scatter(
        self,
        num_rows: int = 3,
        num_cols: int = 3,
        fig_width: int = 10,
        fig_height: int = 9.5,
    ) -> None:
        """
        Show the relationship between TF and gene changes through scatter plot.

        Parameters
        ----------
        num_rows
            The number of rows in the graph.
            (Default: 3)
        num_cols
            The number of columns in the graph.
            (Default: 3)
        fig_width
            The width of the image.
            (Default: 10)
        fig_height
            The height of the image.
            (Default: 9.5)
        """
        fig = plt.figure(figsize=(fig_width,fig_height))

        for i in range(num_rows*num_cols):
            row_idx=i
            row=self.network_df.iloc[row_idx]
            gene_name=row['gene']
            TF_name=row['TF']

            ax = fig.add_subplot(num_rows, num_cols, i+1)
            self.lm(gene_name,TF_name,ax)
            # subplt.imshow(plot)
            # subplt.axis('off')
        # fig.savefig('output.png',dpi=300)
        # plt.show()
        return fig

    def generate_data_2_time(self, cell_generate_per_time, cell_select_per_time):
        """
        Generate data in the 2_time mode.

        Parameters
        ----------
        cell_generate_per_time
            The amount of data generated at each time point.
        cell_select_per_time
            The amount of data randomly selected at each time point.
        """
        # delta_gene = self.adata2.X - self.adata1.X
        delta_gene = self.data2.exp_matrix - self.data1.exp_matrix
        if issparse(delta_gene):
            delta_gene = delta_gene.toarray()

        self.get_one_hot()

        self.input_data = np.array(delta_gene, dtype=np.float32)
        self.output_data = np.array(self.data2.exp_matrix @ self.T.T, dtype=np.float32)

        self.getMetaData(cell_generate_per_time, cell_select_per_time)

        # normalize data
        self.input_data = (self.input_data - self.input_data.mean(axis=0)) / (
            self.input_data.max(axis=0) - self.input_data.min(axis=0)
        )
        self.output_data = (self.output_data - self.output_data.min(axis=0)) / (
            self.output_data.max(axis=0) - self.output_data.min(axis=0)
        )

        # # shuffle the cell order (2_time data has been shuffled in the meta step)
        # permuted_idxs=np.random.permutation(self.input_data.shape[0])
        # self.input_data=nor_input_data[permuted_idxs]
        # self.output_data=nor_output_data[permuted_idxs]

        self.get_dataloader()

    def generate_data_p_time(
        self, cell_divide_per_time, cell_select_per_time, cell_generate_per_time
    ) -> None:
        """
        Generate data in the 2_time mode.

        Parameters
        ----------
        cell_divide_per_time
            The number of divided cells per time point.
        cell_select_per_time
            The amount of data randomly selected at each time point.
        cell_generate_per_time
            The amount of data generated at each time point.
        """
        sub_index = self.sort_idx(cell_divide_per_time)

        self.mean_data = []  # mean expression of genes at each time point
        self.origin_data = []  # time_point * cell * gene expression
        for i in range(len(sub_index)):
            self.origin_data.append(np.array(self.adata[sub_index[i]].X))
            self.mean_data.append(self.adata[sub_index[i]].X.mean(axis=0))
        self.mean_data = np.array(self.mean_data)
        self.origin_data = np.stack(self.origin_data)

        self.get_one_hot()

        input_data = []
        output_data = []
        for i in range(1, len(self.origin_data)):
            for j in range(cell_generate_per_time):
                random_idxs = random.sample(
                    range(len(self.origin_data[i])), cell_select_per_time
                )
                meta_expr = np.array(self.origin_data[i][random_idxs].mean(axis=0))
                delta_gene = meta_expr - self.mean_data[i - 1]
                tf_expr = meta_expr @ self.T.T

                input_data.append(delta_gene)
                output_data.append(tf_expr)
        print(
            f"{(len(self.origin_data)-1)} groups of new data were generated, each with {cell_generate_per_time} meta cells."
        )

        input_data = np.array(input_data, dtype=np.float32)
        output_data = np.array(output_data, dtype=np.float32)

        nor_input_data = (input_data - input_data.mean(axis=0)) / (
            input_data.max(axis=0) - input_data.min(axis=0)
        )
        nor_output_data = (output_data - output_data.min(axis=0)) / (
            output_data.max(axis=0) - output_data.min(axis=0)
        )

        permuted_idxs = np.random.permutation(input_data.shape[0])
        self.input_data = nor_input_data[permuted_idxs]
        self.output_data = nor_output_data[permuted_idxs]

        self.get_dataloader()

    def get_dataloader(self):
        """
        Convert data to dataloader form.
        """
        train_ratio = self.train_ratio
        num_samples = self.input_data.shape[0]
        train_size = int(num_samples * train_ratio)

        train_data_in = torch.from_numpy(self.input_data[:train_size, :])
        train_data_out = torch.from_numpy(self.output_data[:train_size, :])
        test_data_in = torch.from_numpy(self.input_data[train_size:, :])
        test_data_out = torch.from_numpy(self.output_data[train_size:, :])

        train_set = TensorDataset(train_data_in, train_data_out)
        test_set = TensorDataset(test_data_in, test_data_out)
        batch_size = 32
        self.train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.test_dl = DataLoader(test_set, batch_size=batch_size)

    def plot_gene_regulation(self, min_weight, min_node_num, cmap="coolwarm") -> None:
        """
        Draw the gene regulation network graph.

        Parameters
        ----------
        min_weight
            Filter relation pairs whose weight is less than min_weight.
        min_node_num
            Min node numbef of Tfs.
        cmap
            Platte used..
            (Default: 'coolwarm')
        """
        df = self.network_df
        df = df.loc[df["weight"].abs() > min_weight]
        print(f"num of weight pairs after weight filtering: {len(df)}")

        df_TF = pd.Series(df["TF"].value_counts())
        label_name = pd.Series(df_TF[df_TF >= min_node_num].index).to_dict()
        df = df.loc[df["TF"].isin(label_name.values())]
        print(f"num of weight pairs after node_count filtering: {len(df)}")

        G = nx.from_pandas_edgelist(df, "TF", "gene", create_using=nx.Graph())

        nodes = G.nodes()
        degree = G.degree()
        colors = [degree[n] for n in nodes]
        # size = [(degree[n]) for n in nodes]

        pos = nx.kamada_kawai_layout(G)

        betCent = nx.betweenness_centrality(G, normalized=True, endpoints=True)
        node_color = [2000.0 * G.degree(v) for v in G]
        # node_color = [community_index[n] for n in H]
        node_size = 5

        label_name_new = dict(zip(label_name.values(), label_name.values()))

        fig = plt.figure(figsize=(6, 6), dpi=200)
        # nx.draw_networkx(G,pos,alpha = 0.8, node_color = node_color,
        #     node_size = node_size ,font_size = 20, width = 0.4, cmap = cmap,
        #                 with_labels=True, labels=label_name_new,edge_color ='grey')
        nx.draw_networkx_nodes(
            G, pos, node_color=node_color, node_size=node_size, cmap=cmap, alpha=0.8
        )
        nx.draw_networkx_edges(G, pos, alpha=0.1, width=1)
        nx.draw_networkx_labels(
            G,
            pos,
            font_size=8,
            font_color="orange",
            labels=label_name_new,
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "orange", "linewidth": 1},
        )

    def get_one_hot(
        self,
    ) -> None:
        """
        Generate one-hot matrix from TFs to genes.

        Returns
        -------
            One-hot matrix from TFs to genes.
        """
        vectorizer = CountVectorizer(
            vocabulary=self.genes.tolist(), lowercase=False
        )  # lowercase gene names
        self.T = vectorizer.fit_transform(self.tfs).toarray()

    def sort_idx(self, cell_divide_per_time) -> np.array:
        """
        Divide the cells in the pseudo-time series to obtain cell sets of different time segments.

        Parameters
        ----------
        cell_divide_per_time
            The number of divided cells per time point.

        Returns
        -----------
        np.array
            An array to divide cells.
        """
        ptime = self.adata.obs["ptime"]

        ptime_0_idx = ptime[ptime == 0].index
        ptime_1_idx = ptime[ptime == 1].index

        middle_idx = list(
            self.adata.obs.index[len(ptime_0_idx) : len(self.adata) - len(ptime_1_idx)]
        )
        sub_length = cell_divide_per_time
        sub_index = [
            middle_idx[i : i + sub_length]
            for i in range(0, len(middle_idx), sub_length)
        ][:-1]

        print(
            f"{len(self.adata)} cells were divided into {len(sub_index)} groups according to the pseudo-time, with {cell_divide_per_time} cells in each group. The first and last cells are discarded."
        )

        # sub_index.insert(0,list(ptime_0_index)) # insert head indexes
        # sub_index.append(list(ptime_1_index))   # append tail inedexes

        return sub_index

    def train(self, dataloader, model, loss_fn, optimizer):
        """
        Train step.

        Parameters
        ----------
        dataloader
            Dataloader that contains the input and output data.
        model
            Model used to infer the realationship.
        loss_fn
            Loss function.
        optimizer
            The optimizer used to reduce the loss value.

        Returns
        -----------
        float
            Training loss.
        """
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            # 计算预测误差
            pred = model(X)
            loss = loss_fn(pred, y)

            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # if batch % 25 ==0:
            #     loss,current=loss.item(), (batch+1)*len(X)
            #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        return loss.item()

    def test(self, dataloader, model, loss_fn):
        """
        Test step

        Parameters
        ----------
        dataloader
            Dataloader that contains the input and output data.
        model
            Model used to infer the realationship.
        loss_fn
            The optimizer used to reduce the loss value.

        Returns
        -----------
        float
            Testing loss.
        """
        # size=len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
        test_loss /= num_batches

        return test_loss
        print(f"Avg loss: {test_loss:>8f} \n")
