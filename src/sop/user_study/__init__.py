import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from collections import defaultdict
import random
import math
from tqdm import tqdm
import pandas as pd
import os


def get_df(filepaths):
    dfs = []
    for filepath in filepaths:
        df = pd.read_csv(filepath)
        dfs.append(df)
    df = pd.concat(dfs)
    return df

def get_df_dir(filepaths_dir):
    return get_df([os.path.join(filepaths_dir, filepath) for filepath in os.listdir(filepaths_dir)])


def get_bar_plot_with_err_bar(df, group_names, group_column, score_columns, custom_names, name_mapping=None, save_path=None,
                              title=None, xlabel=None, ylabel=None, rotation=0, agg_func='mean'):
    # group_names = metrics
    # group_column = 'Input.metric'
    # score_columns = ['diff', 'abs_diff']
    # custom_names = score_columns

    plt.rcParams.update({
        'font.size': 9,
        'font.family': 'STIXGeneral' #'Times New Roman'
    })  # Set a default font size

    if name_mapping is None:
        name_mapping = {k: k for k in group_names}

    # import pdb; pdb.set_trace()
    filtered_df = df #[group_column].isin(name_mapping.keys())

    # Group by 'method' and calculate mean scores for each question
    if agg_func == 'mean':
        grouped_data = filtered_df.groupby(group_column)[score_columns].mean().T
    elif agg_func == 'median':
        grouped_data = filtered_df.groupby(group_column)[score_columns].median().T
    elif agg_func == 'max':
        grouped_data = filtered_df.groupby(group_column)[score_columns].max().T
    elif agg_func == 'min':
        grouped_data = filtered_df.groupby(group_column)[score_columns].min().T
    elif agg_func == 'sum':
        grouped_data = filtered_df.groupby(group_column)[score_columns].sum().T
    else:
        raise ValueError(f'Unsupported aggregation function: {agg_func}')

    # Rename the methods according to your dictionary
    grouped_data.columns = [name_mapping.get(x, x) for x in grouped_data.columns]

    # Calculate standard deviations using bootstrapping
    stds_dict = defaultdict(dict)
    num_bootstrap = 1000

    for qi in range(len(score_columns)):
        score_name = score_columns[qi]
        means = defaultdict(list)
        grouped_data_dict = filtered_df.groupby(group_column)[score_name].apply(list).to_dict()

        for i in tqdm(range(num_bootstrap)):
            bootstrap_k = len(grouped_data_dict[group_names[0]])
            exp_idxs = random.choices(list(range(bootstrap_k)), k=bootstrap_k)
            for key in name_mapping:
                means[key].append(np.mean([grouped_data_dict[key][idx] for idx in exp_idxs if not math.isnan(grouped_data_dict[key][idx])]))
        for key in name_mapping:
            stds_dict[key][score_name] = np.std(means[key])

    # Convert stds_dict to a DataFrame for easier access
    stds_df = pd.DataFrame(stds_dict).T
    stds_df.columns = score_columns
    stds_df = stds_df.T
    stds_df.columns = [name_mapping.get(x, x) for x in stds_df.columns]
    stds_df.index = custom_names

    # Number of questions
    n_questions = len(grouped_data)

    # Create an array of positions for the bars in a group
    bar_width = 0.7
    spacing = 1  # Space between groups
    index = np.arange(0, n_questions * (bar_width * grouped_data.shape[1] + spacing), bar_width * grouped_data.shape[1] + spacing)

    # Create the plot
    fig, ax = plt.subplots(figsize=(2.5, 1.5))

    # Define a list of colors from a palette
    colors_all = cm.get_cmap('tab20')  # The second argument specifies how many discrete colors to generate
    hatches_all = ['///', '\\\\\\', '---', '++++', 'xxxx', 'oo', '...', '**', '']

    # Generate colors from the colormap
    colors = [colors_all(i) for i in range(len(name_mapping))]
    hatches = [hatches_all[i] for i in range(len(name_mapping))]

    # Adding bars for each method with error bars
    for i, (method, scores) in enumerate(grouped_data.items()):
        color = colors[i % len(colors)]  # Use modulo to cycle through colors if there are more methods than colors
        hatch = hatches[i % len(hatches)]
        yerr = stds_df[method].values  # Get the standard deviations for the current method
        ax.bar(index + i * bar_width, scores, bar_width, label=method, color=color, hatch=hatch, edgecolor='white', yerr=yerr)

    # Adding labels and titles
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None: 
        ax.set_ylabel(ylabel)
    ax.set_xticks(index + bar_width * (grouped_data.shape[1] - 1) / 2)
    ax.set_xticklabels(custom_names, rotation=rotation)
    # ax.set_ylim(0, 5)  # Set y-axis limits to the range of your scores

    legend = ax.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left',
                       labelspacing=0.2, handlelength=1, handletextpad=0.5, 
                       handleheight=0.5, borderpad=0.5)

    # Show the plot
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

    return grouped_data, stds_df