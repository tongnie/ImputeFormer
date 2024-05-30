import os
from datetime import datetime, timedelta
from functools import reduce
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pandas.tseries.frequencies as pd_freq
import tsl
from tqdm import tqdm
from tsl.datasets.prototypes import PandasDataset, DatetimeDataset
from tsl.utils import download_url, extract_zip

from .. import config

START = datetime(2008, 12, 31, 0, 0)
ID_COL = 'id'
TARGET_COL = 'load'
DATETIME_COL = 'datetime'
SAMPLES_PER_DAY = 48
AGG_SCALE = 1000
TEST_LEN = 0.2


def parse_date(date):
    """
    Parses date strings for the irish dataset.

    :param date: timestamp (see dataset description for information)
    :return: datetime
    """
    return START + timedelta(days=date // 100) + timedelta(
        hours=0.5 * (date % 100))


class CEREn(DatetimeDataset):
    url = ""  # request url at https://www.ucd.ie/issda/data/commissionforenergyregulationcer/

    similarity_options = {'correntropy', 'pearson'}
    default_freq = '30T'

    def __init__(self,
                 root=None,
                 freq=None,
                 num_sensors=435):
        # set root path
        self.root = config['data_dir'] + '/cer_e/' if root is None else root
        self.num_sensors = num_sensors
        # load dataset
        df, mask = self.load()
        super().__init__(df,
                         mask=mask,
                         freq=freq,
                         similarity_score='correntropy',
                         temporal_aggregation='sum',
                         spatial_aggregation='sum',
                         name='CEREn')

    @property
    def raw_file_names(self):
        return [f'File{i}.txt.zip' for i in range(1, 7)] + \
               ['allocations.xlsx', 'manifest.docx']

    @property
    def required_file_names(self):
        return ['cer_en.h5', 'allocations.xlsx', 'manifest.docx']

    # def download(self) -> None:
    #     path = download_url(self.url, self.root_dir)
    #     extract_zip(path, self.root_dir)
    #     os.unlink(path)
    #     downloaded_folder = os.path.join(self.root_dir, 'irish')
    #     # move files to root folder
    #     for file in os.listdir(downloaded_folder):
    #         if file in self.raw_file_names:
    #             os.rename(os.path.join(downloaded_folder, file),
    #                       os.path.join(self.root_dir, file))
    #     self.clean_root_dir()

    # def build(self):
    #     # self.maybe_download()
    #     # Build dataset
    #     tsl.logger.info("Building the dataset...")
    #     dfs = []
    #     # read csv from zip files
    #     for filepath in tqdm(filter(lambda x: '.zip' in x,
    #                                 os.listdir(self.root_dir))):
    #         filepath = os.path.join(self.root_dir, filepath)
    #         zip = ZipFile(filepath)
    #         ifile = zip.open(zip.infolist()[0])
    #         data = pd.read_csv(ifile, sep=" ", header=None,
    #                            names=[ID_COL, DATETIME_COL, TARGET_COL])
    #         data = data.apply(pd.to_numeric)
    #         data = pd.pivot_table(data, values=TARGET_COL, index=[DATETIME_COL],
    #                               columns=[ID_COL])
    #         dfs.append(data)
    #     # merge dfs
    #     df = reduce(lambda left, right: pd.merge(left, right, on=DATETIME_COL),
    #                 dfs)
    #
    #     # parse datetime index
    #     df = df.reset_index()
    #     ts = df[DATETIME_COL].values % 100
    #     # remove inconsistent timestamps
    #     df = df[(ts > 0) & (ts <= SAMPLES_PER_DAY)]
    #     index = pd.to_datetime(df[DATETIME_COL].apply(parse_date))
    #     df.loc[:, DATETIME_COL] = index
    #     df = df.drop_duplicates(DATETIME_COL)
    #     df = df.set_index(DATETIME_COL).astype('float32')
    #
    #     # save df
    #     path = os.path.join(self.root_dir, 'cer_en.h5')
    #     df.to_hdf(path, key='data', complevel=3)
    #     # self.clean_downloads()
    #
    #     return df

    def load(self):
        df = self.load_raw()
        tsl.logger.info('Loaded raw dataset.')
        # Fix missing timestamps
        df = df.asfreq(self.default_freq)
        mask = ~np.isnan(df.values)
        df = df.fillna(0.)
        return df, mask

    # def load_raw(self) -> pd.DataFrame:
    #     # self.maybe_build()
    #     return pd.read_hdf(self.required_files_paths[0])

    def load_raw(self) -> pd.DataFrame:
        data_df = pd.read_hdf(self.required_files_paths[0])
        data_df = data_df.iloc[:, :self.num_sensors]  # filter subset
        return data_df

    def aggregate(self, node_index=None, aggr: str = None,
                  mask_tolerance: float = 0.):
        aggregate = super(CEREn, self).aggregate(node_index, aggr,
                                                 mask_tolerance)
        return aggregate / AGG_SCALE

    def compute_similarity(self, method: str, gamma=0.05,
                           train_slice=None, mask=None,
                           **kwargs):
        train_df = self.dataframe()
        if mask is None:
            mask = self.mask
        train_df = train_df * mask[..., -1]
        if train_slice is not None:
            train_df = self.dataframe().iloc[train_slice]
        if method == 'pearson':
            tot = train_df.mean(1).to_frame()
            bias = tot.groupby([tot.index.weekday,
                                tot.index.hour,
                                tot.index.minute]).transform(np.nanmean).values
            scale = train_df.values.std(0, keepdims=True)
            train_df = train_df - bias * scale
            # sim = pearson_sim_matrix(train_df.values.T)
            sim = np.corrcoef(train_df.values, rowvar=False)
            sim = sim[:self.num_sensors, :self.num_sensors]
        elif method == 'correntropy':
            from sklearn.metrics.pairwise import rbf_kernel
            x = train_df.values
            x = (x - x.mean()) / x.std()
            # one week
            period = pd_freq.to_offset('7D').nanos // self.freq.nanos
            chunks = range(period, len(x), period)
            sim = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
            for i in tqdm(chunks, desc="Computing correntropy for every week"):
                xi = x[i - period:i].T
                sim += rbf_kernel(xi, gamma=gamma)
            sim /= len(range(period, len(x), period))
            sim = sim[:self.num_sensors,:self.num_sensors]
        else:
            raise NotImplementedError
        return sim