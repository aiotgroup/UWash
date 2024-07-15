class DatasetConfig():
    _defaults = {
        # 分类个数
        "n_classes": 10,
        # 数据集根目录
        # "base_path": "/home/wuxilei/data/WatchDataProcess/",
        "base_path": "D:/WatchCollectData/",
        # 已生成好的数据源目录
        "filename_datasource": "datasource_64/",
        "filename_labels": "label_action.json",
        "filename_locations": "location_info.json",
        "filename_persons": "offsets.json",
        # 序列长度
        "seq_len": 64,
    }

    @classmethod
    def get(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
