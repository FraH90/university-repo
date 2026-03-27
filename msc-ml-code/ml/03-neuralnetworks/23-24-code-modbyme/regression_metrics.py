import numpy as np


class RegressionMetrics:

    def __init__(self, y, pred):
        self.y = y
        self.pred = pred
        self.error = self.pred - self.y

    def compute_errors(self):
        return {
            "mae": self.mae(),
            "mse": self.mse(),
            "rmse": self.rmse(),
            "mape": self.mape(),
            "mpe": self.mpe(),
            "r2_score": self.r2_score(),
        }

    def mae(self):
        return np.average(np.abs(self.error))

    def mse(self):
        return np.average(self.error ** 2)

    def rmse(self):
        return np.sqrt(self.mse())

    def mape(self):
        return np.average(np.abs(self.error / self.y))

    def mpe(self):
        return np.average(self.error / self.y) * 100

    def r2_score(self):
        ssr = np.sum(self.error ** 2)
        sst = np.sum((self.y - self.y.mean()) ** 2)

        return 1 - (ssr / sst)
