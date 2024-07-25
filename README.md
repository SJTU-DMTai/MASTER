# Readme 
This is the official code and supplementary materials for our AAAI-2024 paper: **MASTER: Market-Guided Stock Transformer for Stock Price Forecasting**. [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/27767)  [[ArXiv preprint]](https://arxiv.org/abs/2312.15235) 

MASTER is a stock transformer for stock price forecasting, which models the momentary and cross-time stock correlation and guides feature selection with market information.

![MASTER framework](framework.png)

Our original experiments were conducted in a complex business codebase developed based on Qlib. The original code is confidential and exhaustive. In order to enable anyone to quickly use MASTER and reproduce the paper's results, here we publish our well-processed data and core code. 

## Usage
1. Install dependencies.
- pandas == 1.5.3
- torch == 1.11.0

2. Install [Qlib](https://github.com/microsoft/qlib). We have minimized the reliance on Qlib, and you can simply install it by
- <code>pip install pyqlib </code> (Pip installation only supports python 3.7 and 3.8, please refer to its Readme.md.)
- pylib == 0.9.1.99

3. Download data from one of the following links (the data files are the same) and unpack it into <code> data/ </code>
- [OneDrive link](https://1drv.ms/f/c/652674690cc447e6/Eu8Kxv4xxTFMtDQqTW0IU0UB8rnpjACA5twMi8BA_PfbSA?e=ooc0za)
- [MEGA link](https://mega.nz/file/4OE0jK4I#h-LG7OjFnncbL_YGoSx5c0W604OdFMgALTYFcoDvgfw)
- [Baidu link](https://pan.baidu.com/s/1Wv_nzmw6vlexqZinV_zLtA?pwd=hrrl). 

5. Run main.py.

6. We provide two trained models: <code> model/csi300master_0.pkl, model/csi800master_0.pkl</code>

## Dataset
### Form
The downloaded data is split into training, validation, and test sets, with two stock universes. Note the csi300 data is a subset of the csi800 data.
You can use the following code to investigate the **datetime, instrument, and feature formulation**.
```python
with open(f'data/csi300/csi300_dl_train.pkl', 'rb') as f:
    dl_train = pickle.load(f)
    dl_train.data # a Pandas dataframe
```
In our code, the data will be gathered chronically and then grouped by prediction dates. the <code> data </code> iterated by the data loader is of shape (N, T, F), where:
- N - number of stocks. For CSI300, N is around 300 on each prediction date; For CSI800, N is around 800 on each prediction date.
- T - length of lookback_window, T=8.
- F - 222 in total, including 158 factors, 63 market information, and 1 label.        

### Market information
For convenient reference, we extract and organize market information from the published data into <code>data/csi_market_information.csv</code>.
You can check the **datetime and feature formulation** in the file. Note that m is shared by all stocks.
The market data is generated by the following pseudo-code.

```python
m = []
for S in csi300, csi500, csi800:
  m += [market_index(S,-1)]
  for d in [5, 10, 20, 30, 60]:
    m += [historical_market_index_mean(S, d), historical_market_index_std(S, d)]
    m += [historical_amount_mean(S, d), historical_amount_std(S, d)]
```

### Preprocessing
The published data went through the following necessary preprocessing. 
1. Drop NA features, and perform robust daily Z-score normalization on each feature dimension.
2. Drop NA labels and 5% of the most extreme labels, and perform **daily Z-score normalization** on labels. 
- Daily Z-score normalization is a common practice in Qlib to standardize the labels for stock price forecasting. To mitigate the difference between a normal distribution and groundtruth distribution, we filtered out 5\% of most extreme labels in training. Note that the reported RankIC compares the output ranking with the groundtruth, whose value is not affected by the label normalization.

## An Alternative Qlib implementation
We are happy to hear that MASTER has been integrated into the open-sourced Qlib framework at this [repo](https://github.com/SJTU-Quant/qlib/tree/main/examples/benchmarks/MASTER). We thank [LIU, Qiaoan](https://github.com/zhiyuan5986) and [ZHAO, Lifan](https://github.com/MogicianXD) for their contributions and please also give credits to the new repo if you use it. 

As a brief introduction to the new version, with the Qlib framework, you can
- report AR, IR, and more portfolio-based metrics,
- modify experiment configuration with .yaml files,
- compare with various models from the Qlib examples collection,
- benefit from other merits of Qlib.

In the meantime, please note that
- The new version utilizes **a different data source** published by Qlib, which covers a different timespan. The new data source is considered **logically equal** to our published data but may differ in values.
- :fire:[Add new notice] The new version uses **stock universe CSI300 & CSI500**, because qlib does **not** include a CSI800 dataset. Correspondingly, **the representative indices to construct market information are different**, it uses CSI100, CSI300, and CSI500, which is different from CSI300, CSI500, and CSI800 as in this repo.
- The new version does **not** include the 'DropExtremeLabel' operation in data preprocessing but also reports decent performance.

## Cite
If you use the data or the code, please cite our work! :smile:
```latex
@inproceedings{li2024master,
  title={Master: Market-guided stock transformer for stock price forecasting},
  author={Li, Tong and Liu, Zhaoyang and Shen, Yanyan and Wang, Xue and Chen, Haokun and Huang, Sen},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={1},
  pages={162--170},
  year={2024}
}
```


