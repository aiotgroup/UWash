class TrainConfig():
    _defaults = {
        # 训练轮数
        "num_epochs": 300,
        # MiniBatch数量
        "batch_size": 1024 * 4,
        # 训练模式
        "mode": "normal",
        # 训练模型下选取测试集index
        "index": 0,
        # 是否使用GPU
        "cuda": True,
        # 初始学习率
        "init_lr": 1e-3,
        # 动量
        "momentum": 0.99,
        # 学习率下降轮数
        "lr_down_milestones": [200, 300, 400, 500, 600, 700, 800],
        # 学习率下降比例
        "lr_down_ratio": 0.1,
        # 检查点路径
        "check_point_path": "./checkpoint/",
        # 选择loss训练
        "loss": "CEL",
    }

    @classmethod
    def get(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
