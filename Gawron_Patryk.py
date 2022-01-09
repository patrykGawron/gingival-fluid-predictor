import argparse
import pathlib

import pandas as pd

from processing.utils import perform_processing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()

    input_file = pathlib.Path(args.input_file)
    results_file = pathlib.Path(args.results_file)

    data = pd.read_csv(input_file, sep='\t')

    column_names = ['16-B', '16-P', '11-B', '11-P', '24-B', '24-P', '36-B', '36-P', '31-B', '31-P', '44-B', '44-P']
    gt_data = data[column_names]
    input_data = data.drop(columns=column_names)

    predicted_data = perform_processing(input_data)
    print(predicted_data.head())

    predicted_data.to_csv(results_file, sep='\t', encoding='utf-8')


if __name__ == '__main__':
    main()