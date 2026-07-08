import numpy as np


class KukaPreprocessor:
    def __init__(
        self,
        normalization=True,
        remove_constant=False,
        constant_eps=1e-8,
        corr_threshold=None,
        clip_percentiles=None,
    ):
        self.normalization = normalization
        self.remove_constant = remove_constant
        self.constant_eps = constant_eps
        self.corr_threshold = corr_threshold
        self.clip_percentiles = clip_percentiles

    def fit(self, x_train):
        x = np.asarray(x_train)

        # 1. constant features
        if self.remove_constant:
            std = x.std(axis=0)
            self.constant_mask_ = std > self.constant_eps
        else:
            self.constant_mask_ = np.ones(x.shape[1], dtype=bool)

        x = x[:, self.constant_mask_]

        # 2. highly correlated features
        if self.corr_threshold is not None:
            corr = np.corrcoef(x, rowvar=False)
            corr = np.nan_to_num(np.abs(corr), nan=0.0)

            upper = np.triu(corr, k=1)

            # dropping j_th feature if correlated with previous i_th
            drop = np.any(upper > self.corr_threshold, axis=0)
            self.corr_mask_ = ~drop
        else:
            self.corr_mask_ = np.ones(x.shape[1], dtype=bool)

        x = x[:, self.corr_mask_]

        # 3. clipping
        if self.clip_percentiles is not None:
            low, high = self.clip_percentiles
            self.q_low_ = np.percentile(x, low, axis=0)
            self.q_high_ = np.percentile(x, high, axis=0)
            x_for_stats = np.clip(x, self.q_low_, self.q_high_)
        else:
            self.q_low_ = None
            self.q_high_ = None
            x_for_stats = x

        # 4. normalization
        if self.normalization:
            self.min = x_for_stats.min(axis=0, keepdims=True)
            self.max = x_for_stats.max(axis=0, keepdims=True)
            self.range = self.max - self.min
            self.range[self.range < self.constant_eps] = 1.0

        else:
            self.min = None
            self.max = None
            self.range = None

        return self

    def transform(self, x):
        x = np.asarray(x)

        x = x[:, self.constant_mask_]
        x = x[:, self.corr_mask_]

        if self.clip_percentiles is not None:
            x = np.clip(x, self.q_low_, self.q_high_)

        if self.normalization:
            x = (x - self.min) / self.range

        return x.astype(np.float32)

    def fit_transform(self, x_train):
        return self.fit(x_train).transform(x_train)
