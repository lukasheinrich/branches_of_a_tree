import numpy as np
import pandas as pd

COLORS = {
    'stad': 'maroon',
    'scorebase': 'darkorange',
    'numeric': 'forestgreen',
    'score': 'steelblue'
}

def array_mean_and_quantiles(array, window = 5):
    mean = pd.DataFrame(np.mean(array, axis=1)).rolling(window).mean().to_numpy()[:,0]
    pup,pmn,pdn = [
        pd.DataFrame(np.quantile(array,q, axis=1)).rolling(window).mean().to_numpy()[:,0]
        for q in [.10,.50,.90]
    ]
    return mean,(pup,pmn,pdn)

