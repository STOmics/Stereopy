<table>
    <tr bgcolor=#CDC9C9>
        <td>完成时间</td>
        <td></td>
        <td>数据质控</td>
        <td>细胞识别</td>
        <td>空间特异性</td>
        <td>组织边界</td>
        <td>细胞间相互作用</td>
        <td>基因互作</td>
        <td>细胞命运</td>
        <td>可视化</td>
    </tr>
    <tr>
        <td rowspan="4">已完成</td>
        <td rowspan="3">集成</td>
        <td>
            信息统计
            <ul>
                <li>total_count</li>
                <li>n_gene_by_count</li>
                <li>pct_mt_gene</li>
            </ul>
        </td>
        <td>
            降维
            <ul>
                <li>umap</li>
                <li>tsen</li>
                <li>pca</li>
                <li>low variance</li>
                <li>factor analysis</li>
            </ul>
        </td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>
            静态图
            <ul>
                <li>质控小提琴图</li>
                <li>散点图</li>
                <li>热图</li>
                <li>聚类空间分布图</li>
                <li>marker gene 碎石图</li>
                <li>gene pattern空间分布图</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>过滤</td>
        <td>
            聚类
            <ul>
                <li>leiden</li>
                <li>louvain</li>
            </ul>
        </td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>
            标准化
            <ul>
                <li>normalize_total</li>
                <li>quantile</li>
            </ul>
        </td>
        <td>
            find marker gene
            <ul>
                <li>T test</li>
                <li>wilcoxon</li>
                <li>spatial lag model</li>
            </ul>
        </td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>自研</td>
        <td>
            标准化
            <ul>
                <li>zscore_disksmooth</li>
            </ul>
        </td>
        <td>
            直接注释<sup>1</sup>（RF）
            <ul>
                <li>pearson</li>
                <li>spearmanr</li>
            </ul>
        </td>
        <td>gene pattern</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr bgcolor=#CDC1C5>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="4">Q3</td>
        <td rowspan="2">集成</td>
        <td>
            <ul>
                去批次效应
                <li>复现seurat？</li>
                <li>可提需求</li>
            </ul>
        </td>
        <td>
            <ul>
                find marker<sup>2</sup>(Giotto 待确认)
                <li>scran</li>
                <li>mask</li>
                <li>gini</li>
            </ul>
        </td>
        <td>两区域pathway富集分析</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>静态图：可提需求</td>
    </tr>
    <tr>
        <td>
            <ul>
                cell bin统计
                <li>可提需求</li>
            </ul>
        </td>
        <td>
            <ul>
                注释<sup>2</sup>(Giotto 待确认)
                <li>PAGE</li>
                <li>hypergeometric</li>
                <li>RANK</li>
            </ul>
        </td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">自研</td>
        <td></td>
        <td>cell bin注释</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>交互式可视化：可提需求</td>
    </tr>
    <tr>
        <td></td>
        <td>聚类新算法（白勇，待确认）</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr bgcolor=#CDC1C5>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="4">Q4</td>
        <td rowspan="2">集成</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">自研</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr bgcolor=#CDC1C5>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td colspan="10">参考文献：</td>
    </tr>
    <tr>
        <td colspan="10">1. Ruben Dries, et, al. a toolbox for integrative analysis and visualization of spatial expression data. Genome Biology, </td>
    </tr>
    <tr>
        <td colspan="10">2. Rui Hou, et, al. scMatch: a single-cell gene expression profile annotation tool using reference datasets. Bioinformatics, 2019.</td>
    </tr>
    <tr>
        <td colspan="10">3. </td>
    </tr>
    <tr>
        <td colspan="10">4. </td>
    </tr>
    
    
    
</table>