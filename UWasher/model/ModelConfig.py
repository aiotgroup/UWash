class ModelConfig():
    _defaults = {
        # 分类个数
        "n_classes": 10,
        # 序列长度
        "seq_len": 64,
        # 金字塔池化参数
        "pool_sizes": [8, 4, 2],
        # 初始通道数
        "init_channels": 8,
        # 底层通道数
        "bottom_channels": 16,
        # 多通道输入数
        "n_sensors": 2,
        # 加速度和陀螺仪传感器x,y,z共3轴
        "n_axis": 3,
    }

    @classmethod
    def get(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
