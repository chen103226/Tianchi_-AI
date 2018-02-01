import numpy as np
import pandas as pd
import xlrd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing



#train_df: 训练数据的DataFrame格式 k:缺失值个数
#返回缺失值超过k个的columns
def col_missing(train_df, k):

    col_missing_df = train_df.isnull().sum(axis=0).reset_index()
    col_missing_df.columns = ['col','missing_count']
    col_missing_df = col_missing_df.sort_values(by='missing_count')
    k_missing_col = col_missing_df[col_missing_df.missing_count >= k].\
                        col.values
    return k_missing_col

#删除重复的列
def col_deoverlapping(values_df):

    deoverlap_col = values_df.columns[~values_df.T.duplicated()]
    #col = np.append(object_col, deoverlap_col)
    df_deoverlapping = values_df[deoverlap_col]

    return df_deoverlapping

#获取 float
def obtain_x(train_df,xtype):
    dtype_df = train_df.dtypes.reset_index()
    dtype_df.columns = ['col','type']
    return dtype_df[dtype_df.type==xtype].col.values

#获取时间列
def date_cols(values_df):
    values_date_col = []
    values_col = values_df.columns
    #对每个数值类型的列，判断是否为日期，返回为日期的列名
    for col in values_col:
        if values_df[col].max() > 1e7:
            values_date_col.append(col)
    return values_date_col


#填充缺失值
def fill_nan(values_df):

    #填充缺失值
    values_df.fillna(values_df.median(),inplace=True)
    return values_df



#获取特征值相同的列
def values_uniq(train_df):
    float64_col = obtain_x(train_df,'float64')
    int64_col = obtain_x(train_df,'int64')
    values_col = np.append(float64_col,int64_col)
    values_df = train_df[values_col]
    uniq_col = []

    for col in tqdm(values_col):
        uniq = values_df[col].unique()
        if len(uniq) == 1:
            uniq_col.append(col)
    return uniq_col

#
def cal_corrcoef(values_df, corr = 0.15):
    corr_values = []
    values_col = values_df.columns
    values_col = values_col.tolist()

    #提取出目标值Y
    y_train = values_df.Value.values
    #y_label = values_df.label.values
    values_col.remove('Value')
    #values_col.remove('label')
    for col in values_col:
        corr_values.append(abs(np.corrcoef(values_df[col].values,y_train)[0,1]))
    corr_df = pd.DataFrame({'col':values_col,'corr_value':corr_values})
    corr_df = corr_df.sort_values(by='corr_value',ascending=False)

    corr02 = corr_df[corr_df.corr_value >= corr]
    corr02_col = corr02['col'].values.tolist()
    decorr_train= values_df[corr02_col]
    decorr_train['Value'] = y_train
    #decorr_train['label'] = y_label
    return decorr_train