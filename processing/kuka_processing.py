import numpy as np

class KukaPreprocessor:
    def __init__(
        self,
        standardize=True,
        remove_constant=False,
        constant_eps=1e-8,
        corr_threshold=None,
        clip_percentiles=None,
    ):
        self.standardize = standardize
        self.remove_constant = remove_constant
        self.constant_eps = constant_eps
        self.corr_threshold = corr_threshold
        self.clip_percentiles = clip_percentiles

    def fit(self, x_train):
        x = np.asarray(x_train)

        # 1. feature costanti, solo da train normal
        if self.remove_constant:
            std = x.std(axis=0)
            self.constant_mask_ = std > self.constant_eps
        else:
            self.constant_mask_ = np.ones(x.shape[1], dtype=bool)

        x = x[:, self.constant_mask_]

        # 2. feature altamente correlate, solo da train normal
        if self.corr_threshold is not None:
            corr = np.corrcoef(x, rowvar=False)
            corr = np.nan_to_num(np.abs(corr), nan=0.0)

            upper = np.triu(corr, k=1)

            # drop della feature j se è correlata con una feature precedente i
            drop = np.any(upper > self.corr_threshold, axis=0)
            self.corr_mask_ = ~drop
        else:
            self.corr_mask_ = np.ones(x.shape[1], dtype=bool)

        x = x[:, self.corr_mask_]

        # 3. clipping opzionale, sempre stimato solo da train normal
        if self.clip_percentiles is not None:
            low, high = self.clip_percentiles
            self.q_low_ = np.percentile(x, low, axis=0)
            self.q_high_ = np.percentile(x, high, axis=0)
            x_for_stats = np.clip(x, self.q_low_, self.q_high_)
        else:
            self.q_low_ = None
            self.q_high_ = None
            x_for_stats = x

        # 4. standardizzazione, solo da train normal
        if self.standardize:
            self.mean_ = x_for_stats.mean(axis=0, keepdims=True)
            self.std_ = x_for_stats.std(axis=0, keepdims=True)
            self.std_[self.std_ < self.constant_eps] = 1.0
        else:
            self.mean_ = None
            self.std_ = None

        return self

    def transform(self, x):
        x = np.asarray(x)

        x = x[:, self.constant_mask_]
        x = x[:, self.corr_mask_]

        if self.clip_percentiles is not None:
            x = np.clip(x, self.q_low_, self.q_high_)

        if self.standardize:
            x = (x - self.mean_) / self.std_

        return x.astype(np.float32)

    def fit_transform(self, x_train):
        return self.fit(x_train).transform(x_train)