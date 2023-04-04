from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import random

if __name__ == '__main__':
    dataset = sys.argv[1]
    percentage = float(sys.argv[2])

    data = pd.read_csv(dataset)

    # Number of rows to select 
    num_rows = int(len(data) * percentage)

    # Generate a list of random indices
    random_indices = random.sample(range(0, len(data)), num_rows)

    # Select the rows with the corresponding indices
    random_rows = data.iloc[random_indices]

    # Create a new DataFrame with the selected rows
    data_set = pd.DataFrame(random_rows)
    data_set = data.dropna()

    X = MinMaxScaler().fit_transform(data_set[['cwc_min', 'cwc_max', 'csc_min', 'csc_max' , 'ctc_min' , 'ctc_max' , 'last_word_eq', 'first_word_eq' , 'abs_len_diff' , 'mean_len' , 'token_set_ratio' , 'token_sort_ratio' ,  'fuzz_ratio' , 'fuzz_partial_ratio' , 'longest_substr_ratio']])
    y = data_set['is_duplicate'].values  
    tsne2d = TSNE(
        n_components=2,
        init='random', # pca
        random_state=101,
        method='barnes_hut',
        n_iter=1000,
        verbose=2,
        angle=0.5
    ).fit_transform(X)
    x_df = pd.DataFrame({'x':tsne2d[:,0], 'y':tsne2d[:,1] ,'label':y})

    # draw the plot in appropriate place in the grid
    sns.lmplot(data_set=x_df, x='x', y='y', hue='label', fit_reg=False, palette="Set1",markers=['s','o'])
    plt.show()