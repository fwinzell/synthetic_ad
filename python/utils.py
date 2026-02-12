import numpy as np
from itertools import product
import matplotlib.pyplot as plt

from collections import Counter
from scipy.stats import permutation_test, ks_2samp

class ConfusionVarianceMatrix:
    def __init__(self, cm_array, display_labels=None):
        self.cm_array = cm_array
        self.display_labels = display_labels

    def plot(self, *, cmap="viridis"):

        fig, ax = plt.subplots()

        cm = np.round(np.mean(self.cm_array, axis=0), 1)
        cm_std = np.round(np.std(self.cm_array, axis=0), 1)
        n_classes = cm.shape[0]

        im_kw = dict(interpolation="none", cmap=cmap)

        self.im_ = ax.imshow(cm, **im_kw)
        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(1.0)

        self.text_ = np.empty_like(cm, dtype=object)

        # print text with appropriate color depending on background
        thresh = (cm.max() + cm.min()) / 2.0

        for i, j in product(range(n_classes), range(n_classes)):
            color = cmap_max if cm[i, j] < thresh else cmap_min

            text_cm = f'{cm[i,j]} \n ({cm_std[i,j]})'#format(cm[i, j], ".2g")
            text_kwargs = dict(ha="center", va="center", color=color)
            
            self.text_[i, j] = ax.text(j, i, text_cm, **text_kwargs)

        if self.display_labels is None:
            display_labels = np.arange(n_classes)
        else:
            display_labels = self.display_labels

        #fig.colorbar(self.im_, ax=ax)
        ax.set(
            xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=display_labels,
            yticklabels=display_labels,
            ylabel="True label",
            xlabel="Predicted label",
            facecolor="black",
        )
        ax.grid(False)
        ax.tick_params(axis='both', which='both', length=5, color='black')

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation='horizontal')

        # Set the edge color and line width of the plot
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)

        # Add a border around the colorbar
        cbar = fig.colorbar(self.im_, ax=ax)
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(1)

        self.figure_ = fig
        self.ax_ = ax

        plt.show()
    

def _total_variation_distance(x,y):
    """Function for calculating the TVD (KS statistic equivalent)
    
    Args:
        x (array-like): Real data
        y (array-like): Synthetic data
    
    Returns:
        float : Total variation distance
    
    Example:
        >>> _total_variation_distance([1,2,3,4,5],[1,2,3,4,5])
        0.0
    """
    X, Y = Counter(x), Counter(y)
    merged = X + Y

    return np.round(0.5*sum([abs(X[key]/len(x)-Y[key]/len(y)) for key in merged.keys()]),4)

def _discrete_ks(x, y, n_perms=1000):
    """Function for doing permutation test of discrete values in the KS test
    
    Args:
        x (array-like): Real data
        y (array-like): Synthetic data
        n_perms (int): Number of permutations
    
    Returns:
        float : KS statistic
        float : p-value
    
    Example:
        >>> _discrete_ks([1,2,3,4,5],[1,2,3,4,5])
        (0.0, 1.0)
    """
    res = permutation_test((x, y), _total_variation_distance, n_resamples=n_perms, vectorized=False, permutation_type='independent', alternative='greater')

    return res.statistic, res.pvalue


# Kolmogorov-Smirnov Test Metric borrowed from SynthEval package

class KSMetric(object):
    def __init__(self,
                 sig_lvl=0.05,
                 n_perms=1000,
                 cat_cols=None,
                 num_cols=None
                 ):
        """Initialisation of the Kolmogorov-Smirnov Test Metric class.

        Args:
            sig_lvl (float): Significance level
            n_perms (int): Number of permutations
            cat_cols (list, optional): List of categorical columns. Defaults to None.
            num_cols (list, optional): List of numerical columns. Defaults to None.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
        """
        self.sig_lvl = sig_lvl
        self.n_perms = n_perms
        self.cat_cols = cat_cols if cat_cols is not None else []
        self.num_cols = num_cols if num_cols is not None else []


    def evaluate(self, real_data, synt_data) -> float | dict:
        """Function for executing the Kolmogorov-Smirnov test.

        Args:
            real_data (DataFrame): Real data
            synt_data (DataFrame): Synthetic data
        
        Returns:
            dict: Average KS statistic and standard error of the mean
        """
        n_dists, c_dists = [], []
        pvals = []
        sig_cols = []
        

        for category in real_data.columns:
            R = real_data[category]
            F = synt_data[category]

            if category in self.cat_cols:
                statistic, pvalue = _discrete_ks(F,R,self.n_perms)
                c_dists.append(statistic)
                pvals.append(pvalue)
            else:
                KstestResult = ks_2samp(R,F)
                statistic, pvalue = KstestResult.statistic, KstestResult.pvalue
                n_dists.append(statistic)
                pvals.append(pvalue)
            if pvalue < self.sig_lvl:
                sig_cols.append(category)

        if n_dists == []: avg_ks = np.nan; err_ks = np.nan
        else: avg_ks = np.mean(n_dists); err_ks = np.std(n_dists,ddof=1)/np.sqrt(len(n_dists))

        if c_dists == []: avg_tvd = np.nan; err_tvd = np.nan
        else: avg_tvd = np.mean(c_dists); err_tvd = np.std(c_dists,ddof=1)/np.sqrt(len(c_dists))

        ### Calculate number of significant tests, and fraction of significant tests
        self.results = {'avg stat' : np.mean(n_dists+c_dists), 'stat err' : np.std(n_dists+c_dists,ddof=1)/np.sqrt(len(n_dists+c_dists)),
                        'avg ks'   : avg_ks, 'ks err'   : err_ks,
                        'avg tvd'  : avg_tvd, 'tvd err'  : err_tvd,
                        'avg pval' : np.mean(pvals), 'pval err' : np.std(pvals,ddof=1)/np.sqrt(len(pvals)),
                        'num sigs' : len(sig_cols),
                        'frac sigs': len(sig_cols)/len(pvals),
                        'sigs cols': sig_cols
                        }

        return self.results


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    cm_array = np.random.randint(0, 100, (3, 3, 3))
    cm = ConfusionVarianceMatrix(cm_array, display_labels=["A", "B", "C"])
    #cm.plot()
    #plt.show()

    import pandas as pd
    real = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    fake = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    KST = KSMetric(sig_lvl = 0.1, n_perms=100,cat_cols=['a'], num_cols=['b'])
    res = KST.evaluate(real, fake)
    print(res)

