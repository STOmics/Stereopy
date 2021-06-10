## 框架概述

### 功能

- IO：文件数据读写，生成数据类，目前使用AnnData
- 预处理：对数据类进行质控、过滤、标准化
- 分析：输入数据类以及相应参数设置，运行分析，将分析结果写回数据类
- 可视化：对分析结果进行可视化

### 设计原则

**1.模块化**

考虑到框架的可拓展性，主要设计了五种类，分别为数据类、分析类、并行分析管理类、结果类、可视化类。框架各基类的功能为：

    - 数据类负责输入数据的读写，以及相关属性的获取和设置。
    - 分析类负责对数据进行分析，然后得到分析结果返回。
    - 并行分析管理类负责并行分析的调度管理功能。主要将数据分成多份，并行运行对应分析类，将并行计算的结果进行整合返回。
    - 结果类，即为各分析的结果，主要提供对结果信息的读取。
    - 可视化类，主要负责数据类跟结果类的可视化功能。

**2.类说明**

- **数据抽象类（Data）**

    定义数据类共有的属性和方法，例如read、write、get、set、clear（内存释放）等方法，部分方法子类需重写。
```text
Class Data:
    参数：
    属性:
    方法：
```

- **stereo表达量数据类（StereoExpData）**

    继承Data类，负责处理stereo测序技术的表达量数据，兼容bin和cell bin。
```text
Class StereoExpData:
    参数：
        待补充细化
    属性:
        genes
        cells
        exp_matrix
        position
    方法：
        待补充细化
```

- **10x表达量数据类（TenXExpData）**

    继承Data类，负责处理10x测序技术的表达量数据。
```text
Class TenXExpData:
    参数：
    属性:
        genes
        cells
        exp_matrix
        position
        image
    方法：
```

- **分析基类（ToolBase）**
    接收数据，运行分析，结果返回
```text
Class ToolBase(data, method, output=None):
   参数：待补充细化
     data：数据类
     method: 分析方法，str
     name：分析名字，用于结果落地文件名
   属性：
     self.data
     self.method
     self.name
   方法：待补充细化
     check_param(): 检查参数
     sparse2array(): 将self.exp_matrix稀疏矩阵转换成np.array
     get_params(var_info): 获取参数变量名以及相应的值
     write_res(): 保存分析结果
     merge_res(): merge并行计算的结果
```

- **并行分析管理类（MultiToolManager）**
    接收数据，运行分析，结果返回
```text
Class MultiToolManager(data, tool, *args, **kwargs):
   参数：
     data：数据类
     tool: 分析方法，str

   属性：
     self.data

   方法  
     split_data(): 数据分块
     run(): 并行调度
```

- **结果基类（StereoRes）**
    分析结果基类
```text
Class StereoRes(name='stereo', param=None):
  参数：
    name： 名称， str
  属性：
    self.name
    self.param
  方法：
    __str__(): 返回类的参数信息及结果信息
    __repr__()：打印类的参数信息及结果信息
    get()
```

- **可视化类（Plotting）**
    分析结果基类
```text
Class Plotting(data):
  参数：
    data： 绘图数据， str
  属性：

  方法：

```

### 目录结构
待补充


### 开发说明
1 **书写规范**
  1. 编码要求符合pep8编写规范
  2. 文件命名统一采用小写、下划线；类的命名采用驼峰命名法。
       > eg：文件名：dim_reduce.py   类名：DimReduce
  3. 变量命名采用小写，应该尽可能有意义，避免采用无意的命名，如：a=1， b=2等等
  4. 注释信息要完善，每个函数应该编写其注释信息
  5. 建议使用pycharm进行编辑，编辑器可自动检查编写的代码是否符合规范，且有相关提示。其他编辑器也可以借助安装相关插件来检查。
       ![image](https://raw.githubusercontent.com/BGIResearch/stereopy/dev/docs/source/code_format.png)
     

#### 以下各模块开发说明文档均为v1版的，v2版待更新

--------
2 **tool分析模块开发**
  tool分析模块主要包含两个内容的开发，分别为分析类以及结果类的开发。

2.1 **分析子类编写**
    
2.1.1 子类需继承基类，以及主要重写和实现相关的方法
- 获取参数
- 继承父类构造方法，必须要有的参数：data、method、name
- 重写参数检查方法: self.check_param()
- 实现分析运行方法: self.fit()
        
2.1.2 可自定义分析需要的参数跟方法

**编写示例：**
        
```python
from ..core.tool_base import ToolBase
from ..log_manager import logger


class DimReduce(ToolBase):
    def __init__(self, data: AnnData, method='pca', n_pcs=2, min_variance=0.01, n_iter=250,
                 n_neighbors=5, min_dist=0.3, inplace=False, name='dim_reduce'):
        # 获取参数
        self.params = self.get_params(locals())
        # 继承父类构造方法
        super(DimReduce, self).__init__(data=data, method=method, name=name)
        # 参数检查
        self.check_param()
        # 自定义分析所需属性，结合分析需要
        self.n_pcs = n_pcs
        self.min_variance = min_variance
        self.n_iter = n_iter
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.result = DimReduceResult(name=name, param=self.params)
     
    def check_param(self): #  重写参数检查方法，根据分析所传参数以及其值范围进行检查约束
        """
        Check whether the parameters meet the requirements.
        :return:
        """
        super(DimReduce, self).check_param()
        if self.method.lower() not in ['pca', 'tsen', 'umap', 'factor_analysis', 'low_variance']:
            logger.error(f'{self.method} is out of range, please check.')
            raise ValueError(f'{self.method} is out of range, please check.')
    
    def fit(self, exp_matrix=None):
        # 自定义分析方法实现逻辑
        exp_matrix = exp_matrix if exp_matrix is not None else self.exp_matrix
        if self.method == 'low_variance':
            self.result.x_reduce = low_variance(exp_matrix, self.min_variance)
        elif self.method == 'umap':
            self.result.x_reduce = u_map(exp_matrix, self.n_pcs, self.n_neighbors, self.min_dist)
        else:
            pca_res = pca(exp_matrix, self.n_pcs)
            self.result.x_reduce = pca_res['x_pca']
            self.result.variance_ratio = pca_res['variance_ratio']
            self.result.variance_pca = pca_res['variance']
            self.result.pcs = pca_res['pcs']
         # 通过self.add_result()添加结果回数据类
        self.add_result(result=self.result, key_added=self.name)
``` 
     
2.2 **结果子类编写**
    
- 子类需继承结果基类

- 根据分析结果，自定义该结果类存放的内容以及实现的方法

**编写示例：**
        
```python

```

3 **可视化模块** 

- 绘图基础单元目前包含热图以及散点图
- 每个可视化功能，主要基于分析需展示的结果来决定
- 目前传入数据类主要是anndata，但在获取绘图数据时，应尽可能与anndata抽离，可针对结果类进行数据获取的函数
- 目前采用的面向过程编程，后续看下是否需要改成面向对象

