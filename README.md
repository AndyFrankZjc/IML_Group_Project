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

目前仅对application.csv主表进行了特征处理，还没有进行多表融合

复杂处理包括: 相似来源特征的数学运算、异常值处理、Missing value指示器、稀有类合并等

原主表特征122个，处理后279个
