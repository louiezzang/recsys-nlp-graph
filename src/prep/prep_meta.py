"""
Parses out the metadata from the original csv.
"""
import argparse

import numpy as np
import pandas as pd

from src.utils.logger import logger


def get_category_lvl(category_list: list, lvl=0) -> str:
    try:
        return category_list[lvl]
    except IndexError:
        return 'NA_VALUE'


def get_categories(df: pd.DataFrame) -> pd.DataFrame:
    df['category_lvl_1'] = df['category'].apply(get_category_lvl, args=(0,))
    df['category_lvl_2'] = df['category'].apply(get_category_lvl, args=(1,))
    df['category_lvl_3'] = df['category'].apply(get_category_lvl, args=(2,))
    df['category_lvl_4'] = df['category'].apply(get_category_lvl, args=(3,))
    logger.info('Categories lvl 1 - 4 prepared')

    return df


def get_meta(df: pd.DataFrame) -> pd.DataFrame:
    # Update to reflect if relationship exist
    df['related'] = np.where(df['also_buy'].isnull() | df['also_view'].isnull(), 0, 1)
    df.drop(columns=['also_buy', 'also_view'], inplace=True)

    # Prep categories
    df['category'] = df['category'].apply(eval)
    df['category'] = df['category'].apply(lambda x: x[0] if x else '')  # Get first category only
    df = get_categories(df)

    # Prep title and description
    # TODO: Add cleaning of title and description

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preparing item metadata')
    parser.add_argument('read_path', type=str, help='Path to input csv')
    parser.add_argument('write_path', type=str, help='Path to output csv (of metadata')
    args = parser.parse_args()

    META_COLS = ['asin', 'category', 'title', 'description', 'price', 'brand', 'also_buy', 'also_view']
    df = pd.read_csv(args.read_path, error_bad_lines=False, warn_bad_lines=True,
                     dtype={'asin': 'str', 'title': 'str', 'brand': 'str'},
                     usecols=META_COLS)
    logger.info('DF shape: {}'.format(df.shape))

    meta_df = get_meta(df)

    meta_df.to_csv(args.write_path, index=False)
    logger.info('Csv saved to {}'.format(args.write_path))
