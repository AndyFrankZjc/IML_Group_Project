# IML_Group_Project

数据集太大传不上来，建议下载原数据集后设置以下文件夹：

```text
项目总文件夹
├── raw_data文件夹           # 原数据集的10张csv
├── processed_data文件夹     # 预处理后的csv存放地
├── view_data文件夹          # 取出来csv前几行数据以供浏览
├── feature_select.py        # 特征筛选
├── data_process_train_test.py    # 数据预处理代码
├── make_csv.py              # 功能性代码
└── data_process.py          
```

application_train / test.csv表处理完毕，选了40+特征，one-hot encoding和aggregation后为80+特征

运行步骤：

1. 分别对application_train.csv和application_test.csv运行feature_select.py，生成application_train_selected_features.csv和application_test_selected_features.csv

2. 对application_train_selected_features.csv和application_test_selected_features.csv运行data_process_train_test.py，在最上方的mode中改train / test

3. 得到最终处理后的application_train_processed.csv和application_test_processed.csv


文件说明：

data_process.py是我手工完全修正过的代码，但是对于category转numeric时，由于train和test同一特征的类别数可能不一致，写死的one-hot和frequency规则会出问题，AI写了data_process_train_test.py，将对train的处理规则强加于test上

对特征的处理：

1. 筛选特征，去除重复行

2. 对缺失率 >= 15% 的特征设置MISSING_INDICATOR

3. 对EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3这三个特征做aggregation

4. 构造了6个RATIO（比率）特征

5. 异常值处理winsorization

6. 类别少的one-hot encoding，多的frequency encoding


