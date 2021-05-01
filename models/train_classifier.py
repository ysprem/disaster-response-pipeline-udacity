import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import nltk

# import relevant functions/modules from the nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# import relevant functions/modules from the sklearn

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def load_data(database_filepath):
    """
    Load Data from the Database Function
    
    Arguments:
        database_filepath -> Path to SQLite destination database (e.g. disaster_response_db.db)
    Output:
        X -> a dataframe containing features
        Y -> a dataframe containing labels
        category_names -> List of categories name
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = database_filename.replace(".db","") + "_table"
    df = pd.read_sql_table(table_name,engine)
    
    X = df['message']
    Y = df.iloc[:,4:]
    
    #print(X)
    #print(y.columns)
    category_names = Y.columns # This will be used for visualization purpose
    return X, Y, category_names
  
def tokenize(text):
    """Normalize, replace URLs, tokenize and lemmatize text string
    
    Args:
    text: string. String containing message for processing
       
    Returns:
    clean_tokens: list of strings. List containing processed input
    """
    # Convert text to lowercase and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    #
    url_regex = 'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
    
    detected_urls = re.findall(url_regex, text)
    
    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, "urlplaceholder")

    # Tokenize words
    tokens = word_tokenize(text)
    
    # Stem word tokens and remove stop words
    lemmatizer = nltk.WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    return clean_tokens

def build_model():
    """Build a machine learning pipeline
    
    Args:
    None
       
    Returns:
    cv: gridsearchcv object. Gridsearchcv object that transforms the data, creates the 
    model object and finds the optimal model parameters.
    """
    # Create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize, min_df = 5)),
        ('tfidf', TfidfTransformer(use_idf = True)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 10,
                                                             min_samples_split = 10)))
    ])
    
    # Create parameters dictionary
    parameters = {'vect__min_df': [1, 5],
                  'tfidf__use_idf':[True, False],
                  'clf__estimator__n_estimators':[10, 25], 
                  'clf__estimator__min_samples_split':[2, 5, 10]}
    
    
    # Create grid search object
    cv = GridSearchCV(pipeline, param_grid = parameters, scoring = 'f1_micro', verbose = 10)
    return cv  
  
  
def evaluate_model(model, X_test, Y_test, category_names):
    """Returns test accuracy, precision, recall and F1 score for fitted model
    
    Args:
    model: model object. Fitted model object.
    X_test: dataframe. Dataframe containing test features dataset.
    Y_test: dataframe. Dataframe containing test labels dataset.
    category_names: list of strings. List containing category names.
    
    Returns:
    None
    """
    # Predict labels for test dataset
    Y_pred = model.predict(X_test)

    Y_prediction_test = model.predict(X_test)

    print(classification_report(Y_test.values, Y_prediction_test, target_names=Y.columns.values))
    
def save_model(pipeline, model_filepath):
    """
    Save Pipeline function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        pipeline -> GridSearchCV or Scikit Pipelin object
        pickle_filepath -> destination path to save .pkl file
    
    """
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))  
    
    
def main():
    """
    Train Classifier Main function
    
    This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle
    
    """
    if len(sys.argv) == 3:
        database_filepath, pickle_filepath = sys.argv[1:]
        
        print('Loading data from {} ...'.format(database_filepath))
        
        X, Y, category_names = load_data_from_db(database_filepath)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building the pipeline ...')
        pipeline = build_pipeline()
        
        print('Training the pipeline ...')
        pipeline.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_pipeline(pipeline, X_test, Y_test, category_names)

        print('Saving pipeline to {} ...'.format(pickle_filepath))
        save_model_as_pickle(pipeline, pickle_filepath)

        print('Trained model saved!')

    else:
         print("Please provide the arguments correctly: \nSample Script Execution:\n\
> python train_classifier.py ../data/disaster_response_db.db classifier.pkl \n\
Arguments Description: \n\
1) Path to SQLite destination database (e.g. disaster_response_db.db)\n\
2) Path to pickle file name where ML model needs to be saved (e.g. classifier.pkl")

if __name__ == '__main__':
    main()    
    
