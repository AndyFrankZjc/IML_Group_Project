# IML_Group_Project

数据集太大传不上来，建议下载原数据集后设置以下文件夹：

```text
项目总文件夹
├── raw_data文件夹           # 原数据集的10张csv
├── processed_data文件夹     # 预处理后的csv存放地
├── view_data文件夹          # 取出来csv前几行数据以供浏览
├── feature_select.py        # 特征筛选
├── data_process.py          # 数据预处理代码
├── make_csv.py              # 功能性代码
└── count.py                 # 功能性代码
```

application_train / test.csv表处理完毕，选了46特征，one-hot encoding和aggregation后为68特征

运行步骤：

1. 分别对application_train.csv和application_test.csv运行feature_select.py，生成application_train_selected_features.csv和application_test_selected_features.csv

# 由于训练集和测试集已经提前分好并且特征类别不一致的原因：

2. 对application_train_selected_features.csv和application_test_selected_features.csv运行data_process.py，在第3块：“3. 设置 missing indicator”时，注意是否注释掉“target_col = ["TARGET"]”； 在最后的elif n_unique > 5: 分支下，标注了处理训练集和测试集的代码块，需要进行互斥注释

3. 得到最终处理后的application_train_processed.csv和application_test_processed.csv


文件说明：

feature_select.py和data_process.py是处理数据用的，

make_csv.py可以从大表中取出部分数据来快速观看，

count.py可以统计每个categorical特征中的category数 -> 每个category的样本数和概率，统计numeric特征中的NaN数(缺失率)

对特征的处理：

1. 筛选特征，去除重复行

2. 清洗特征中意义不明的类别

3. 对缺失率 >= 15% 的特征设置MISSING_INDICATOR

4. 对EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3这三个特征做aggregation

5. 构造了6个RATIO（比率）特征

6. 异常值处理winsorization

7. 缺失值填补，categorical型用最高频率，numeric型用中位数

8. categorical转numeric，类别少的one-hot encoding，多的frequency encoding，并将出现频率<2%的合并至稀有类中


