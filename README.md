### TimeMAE

#### Dataset：

https://drive.google.com/drive/folders/128bIAabc5G3RzyYTonPZv8RWttuQ7Roh?usp=drive_link

#### 描述：

利用自监督预训练增强基于深度学习的时间序列模型的表达能力在时间序列分类中越来越流行。尽管已经投入了大量的努力来开发时间序列数据的自监督模型，但我们认为目前的方法还不足以学习最佳的时间序列表示，因为在稀疏的点方向输入单元上只有单向编码。在这项工作中，我们提出了TimeMAE，这是一种新的自监督范式，用于学习基于变压器网络的可转移时间序列表示。

#### 俩阶段训练：

第一阶段自监督的重建

第二阶带标签的监督微调分类层

其中Mydata为自建的序列的数据，用于序列的数据的重建和分类的训练和测试数据集。

har为原项目的公开的数据集

