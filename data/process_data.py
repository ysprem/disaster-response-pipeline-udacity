import pandas as pd
import os
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load and merge messages and categories datasets
    
    Args:
    messages_filepath: string. Filepath for csv file containing messages dataset.
    categories_filepath: string. Filepath for csv file containing categories dataset.
       
    Returns:
    df: dataframe. Dataframe containing merged content of messages and categories datasets.
    """
    
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets
    df = messages.merge(categories, how = 'left', on = ['id'])
    
    
def transform_data(df):
    """Clean dataframe by removing duplicates and converting categories from strings 
    to binary values.
    
    Args:
    df: dataframe. Dataframe containing merged content of messages and categories datasets.
       
    Returns:
    df: dataframe. Dataframe containing cleaned version of input dataframe.
    """
    
    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';',expand=True)
    
    #Fix the categories columns name
    #Select the first row of the categories dataframe
    row = categories.iloc[[1]]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(np.int)
        
        
    # Drop the original categories column from `df`    
    df = df.drop('categories',axis=1)
    
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Given the small number of rows that contain the value 2 in the related field, drop the column
    df = df.drop(['child_alone'],axis=1)
    
    # Remove rows with a related value of 2 from the dataset
    df = df[df.related!=2]
    
    return df
  
def save_data_to_db(df, database_filename):
    """
    Save Data to SQLite Database Function
    
    Arguments:
        df -> Combined data containing messages and categories with categories cleaned up
        database_filename -> Path to SQLite destination database
        
    Returns:
    None
    """    
    engine = create_engine('sqlite:///'+ database_filename)
    table_name = database_filename.replace(".db","") + "_table"
    df.to_sql(table_name, engine, index=False, if_exists='replace')
    
def main():
    """
    Main function which will kick off the data processing functions. There are three primary actions taken by this function:
        1) Load Messages Data with Categories
        2) Clean Categories Data
        3) Save Data to SQLite Database
    """
    
    # Print the system arguments
    # print(sys.argv)
    
    # Execute the ETL pipeline if the count of arguments is matching to 4
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:] # Extract the parameters in relevant variable

        print('Loading messages data from {} ...\nLoading categories data from {} ...'
              .format(messages_filepath, categories_filepath))
        
        df = load_messages_with_categories(messages_filepath, categories_filepath)

        print('Cleaning categories data ...')
        df = clean_categories_data(df)
        
        print('Saving data to SQLite DB : {}'.format(database_filepath))
        save_data_to_db(df, database_filepath)
        
        print('Cleaned data has been saved to database!')
    
    else: # Print the help message so that user can execute the script with correct parameters
        print("Please provide the arguments correctly: \nSample Script Execution:\n\
               python process_data.py disaster_messages.csv disaster_categories.csv disaster_response_db.db \n\
               Arguments Description: \n\
               1) Path to the CSV file containing messages (e.g. disaster_messages.csv)\n\
               2) Path to the CSV file containing categories (e.g. disaster_categories.csv)\n\
               3) Path to SQLite destination database (e.g. disaster_response_db.db)")

if __name__ == '__main__':
    main()
