import copy
import math

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

class Statistics:
    def generate_by_sum(self, list_matrixes):
        valid_matrix = validate_matrixes(list_matrixes)
        result=[]
        for mat in (valid_matrix):
            sumcol1 = 0
            sumcol2 = 0
            for lin in range(len(mat)):
                sumcol1 = sumcol1 + mat[lin, 0]
                sumcol2 = sumcol2 + mat[lin, 1]
            result.append(math.ceil(sumcol2/sumcol1))
        return result

    def generate_by_count(self, list_matrixes):
        valid_matrix = validate_matrixes(list_matrixes)
        result=[]
        for mat in (valid_matrix):
            sumcol1 = 0
            sumcol2 = 0
            for lin in range(len(mat)):
                sumcol1 = sumcol1 + 1
                sumcol2 = sumcol2 + mat[lin, 1]
            result.append(math.ceil(sumcol2/sumcol1))
        return result

    def create_statistical_table(self, metrics, algorithms, data, path, filename):
        statistic_data = {}

        statistic_data['Metric'] = metrics
        dataset =[]
        for i in range(len(data[0])):
            row = []
            for j in range(len(data)):
                row.append(data[j][i])
            dataset.append(row)
        for algo, data in zip(algorithms, dataset):
            statistic_data[algo] = data
        # Create a DataFrame from the data
        df = pd.DataFrame(statistic_data)
        # Set the 'Metric' column as index
        df.set_index('Metric', inplace=True)
        # Create a table plot
        plt.figure(figsize=(4, 4))
        table = plt.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center', cellLoc='center')
        # Styling the table
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.5, 1.5)
        # Remove axis
        plt.axis('off')
        # Set the directory to save the plot
        plot_directory = path
        # Check if the directory exists, if not, create it
        if not os.path.exists(plot_directory):
            os.makedirs(plot_directory)
        # Save the table as a PDF in the specified directory
        plot_file_path = os.path.join(plot_directory, filename)
        plt.savefig(plot_file_path, bbox_inches='tight', pad_inches=0.05)
def validate_matrixes(list_matrixes):
    new_matrixes=[]
    for i in range(len(list_matrixes)):
        mat = copy.deepcopy(list_matrixes[i])
        for lin in range(len(mat)):
            if np.isinf(mat[lin, 1]):
                mat = [1, -1]
                break
        new_matrixes.append(mat)
    return new_matrixes
