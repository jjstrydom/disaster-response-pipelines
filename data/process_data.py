import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads the data from csv files containing the messages and the target categories.
    :param messages_filepath: file path string to the messages csv file
    :param categories_filepath: file path string to the categories csv file
    :return: dataframe containing the messages and the categories
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, left_on="id", right_on="id")
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    # extract a list of new column names for categories.
    category_colnames = categories.iloc[0].str.split('-', expand=True)[0]
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string & convert column from string to numeric
        categories[column] = categories[column].str.split('-', expand=True)[1].astype(int)
    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.merge(df, categories, left_index=True, right_index=True)
    return df

def clean_data(df):
    """
    Takes a dataframe and cleans the data by removing duplicate entries.
    :param df: pandas dataframe containing the data to clean
    :return: pandas dataframe with duplicates removed
    """
    # drop duplicates
    df = df[df.duplicated() == False]
    # TODO: remove outliers
    # There is no data on category child_alone - removing for now to reduce requirements on downstream processes
    # update: rubrik asks for all 36 columns which is silly :(
    # df.drop(columns=['child_alone'], inplace=True)
    return df


def save_data(df, database_filename):
    """
    Saves the data to disk at the specified filepath.
    :param df: pandas dataframe containing the data to save
    :param database_filename: filepath & name string to save the data to
    :return: None (data saved to disk)
    """
    engine = create_engine(database_filename)
    df.to_sql('project_data', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()