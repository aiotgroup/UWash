class EvalConfig():
    _defaults = {
        # MiniBatch数量
        "batch_size": 2048 * 4,
        # 是否使用GPU
        "cuda": True,
        # 模型文件名
        "filename_model": "UWasher-500-epochs.pth",
        # 检查点路径
        "check_point_path": "./checkpoint/",
        # 类别数量
        "n_classes": 10,
        # 训练模式
        "mode": "normal",
        # 训练模型下选取测试集index
        "index": 0,
    }

    @classmethod
    def get(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
