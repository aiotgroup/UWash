class Config():
    _defaults = {
        # 训练轮数
        "epochs": 300,
        # minibatch数量
        "batch_size": 64,
        # 是否使用GPU
        "cuda": True,
        # 学习率
        "learning_rate": 0.001,
        # 动量
        "momentum": 0.99,
        # K折次数
        "k_fold": 10,
        # 分类个数
        "n_classes": 10,
        # 加速度和陀螺仪传感器x,y,z共6轴
        "n_axis": 6,
        # 序列长度
        "seq_len": 64,
        # 序列数据重叠偏差
        "overlap_offset": 32,
        # 是否去掉无用0数据
        "get_rid_of_zero": True,
        # 标注方法：段识别segment，帧识别frame
        "label_method": "segment",
    }

    @classmethod
    def get(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
