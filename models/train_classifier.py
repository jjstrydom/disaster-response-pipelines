import sys
import psutil
from sqlalchemy import create_engine
import pandas as pd
import joblib

import nltk
import ssl
# dealing with certificate issues when trying to download using nltk
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """
    Load the dataset from the local database specified by the database filepath and return the messages as well as their
    targets and target names.
    :param database_filepath: path string to database file location
    :return: tuple of length 3 (X, Y, T) containing
             X = messages,
             Y = targets,
             T = target names
    """
    # load data from database
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table('project_data', con=engine)
    X = df['message']
    Y = df[df.columns[-36:]]
    T = Y.columns
    return X, Y, T


def tokenize(text):
    """
    Take a piece of text and perform NLP (Natural Language Processing) steps. The function tokenizes the message text,
    removes the stopwords, performs lemmatization, and converts the tokens to lowercase.
    :param text: a string of text to process
    :return: list containing clean tokes (strings) generated from the text
    """
    # process text data to tokens
    tokens = word_tokenize(text)
    words = [w for w in tokens if w not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    Construct the sklearn pipeline that vectorizes input messages, performs TF-IDF, and multi output classification
    using a random forest classifier.
    :return: a sklearn pipeline object
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=psutil.cpu_count()), n_jobs=psutil.cpu_count()))
    ])
    return pipeline


def grid_search(pipeline):
    parameters = {
        'clf__estimator__n_estimators': [100, 1000],
        'clf__estimator__criterion': ['gini', 'entropy'],
        'clf__estimator__max_depth': [None, 500],
        'clf__estimator__max_leaf_nodes': [None, 250],
        'clf__estimator__class_weight': [None, 'balanced', 'balanced_subsample']
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv
    # res = cv.fit(X_train, y_train)
    # return res


def print_scores(category, precision, recall, f1score, accuracy, AUC):
    """
    Print the scores nicely formatted so that consecutive prints using this function results in a table structure on
    screen.
    :param category: name of category as a string
    :param precision: precision metric as a float
    :param recall: recall metric as a float
    :param f1score: f1score metric as a float
    :param accuracy: accuracy metric as a float
    :param AUC: AUC metric as a float
    :return: None (prints to screen)
    """
    print(f"{category:23}: {precision:9.3f} {recall:9.3f} {f1score:9.3f} {accuracy:9.3f} {AUC:9.3f}")


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of the model for each category and print it to screen.
    :param model: the trained model to evaluate
    :param X_test: Test messages
    :param Y_test: Test targets
    :param category_names: Category names of the targets present in the data.
    :return: None (prints to screen)
    """
    y_pred_raw = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred_raw, columns=category_names)
    print(f"class                  : precision    recall   f1score  accuracy       AUC")
    for c in category_names:
        precision = precision_score(Y_test[c], y_pred[c], zero_division=0)
        recall = recall_score(Y_test[c], y_pred[c], zero_division=0)
        f1score = f1_score(Y_test[c], y_pred[c], zero_division=0)
        accuracy = accuracy_score(Y_test[c], y_pred[c])
        try:
            AUC = roc_auc_score(Y_test[c], y_pred[c])
        except ValueError:
            AUC = float('nan')
        print_scores(c, precision, recall, f1score, accuracy, AUC)

    precision = precision_score(Y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(Y_test, y_pred, average='weighted', zero_division=0)
    f1score = f1_score(Y_test, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(Y_test, y_pred)
    # remove columns that are made up of only 1 class so we can calculate a valid AUC
    valid_cols = [c for c in Y_test.columns if len(Y_test[c].unique()) == 2]
    AUC = roc_auc_score(Y_test[valid_cols], y_pred[valid_cols])
    print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    print_scores('TOTAL', precision, recall, f1score, accuracy, AUC)


def save_model(model, model_filepath):
    """
    Saves the model to disk.
    :param model: the trained model to save
    :param model_filepath: filepath to save to
    :return: None
    """
    # save model to model filepath
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Building grid search over model parameters...')
        grid = grid_search(model)

        print('Training model...')
        grid.fit(X_train, Y_train)
        # model.fit(X_train, Y_train)
        print(grid.best_params_)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()