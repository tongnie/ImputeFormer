# Interfaces
from .prototypes import Dataset, TabularDataset, DatetimeDataset ,PandasDataset
# Datasets
from .air_quality import AirQuality
from .elergone import Elergone
from .metr_la import MetrLA
from .pems_bay import PemsBay
from .mts_benchmarks import ElectricityBenchmark, TrafficBenchmark, SolarBenchmark, ExchangeBenchmark
from .pems_benchmarks import PeMS03, PeMS04, PeMS07, PeMS08
from .cer_en import CEREn

__all__ = [
    'Dataset',
    'PandasDataset',
    'AirQuality',
    'Elergone',
    'MetrLA',
    'PemsBay',
    'ElectricityBenchmark',
    'TrafficBenchmark',
    'SolarBenchmark',
    'ExchangeBenchmark',
    'CEREn'
]

prototype_classes = __all__[:2]
dataset_classes = __all__[2:]
