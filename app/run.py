import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('project_data', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    category_names = df.columns[4:]
    category_counts = df[category_names].sum()

    label_distribution = df[category_names].sum(axis=1).value_counts().sort_index()

    message_length_distibution = df['message'].str.len().round(-2).value_counts().sort_index()

    graphs = [
            {
                'data': [
                    Bar(
                        x=category_names,
                        y=category_counts
                    )
                ],

                'layout': {
                    'title': 'Distribution of Categories',
                    'yaxis': {
                        'title': "Count"
                    },
                    'xaxis': {
                        'title': "Classes"
                    }
                }
            },
            {
                'data': [
                    Bar(
                        x=label_distribution.index,
                        y=label_distribution.values
                    )
                ],

                'layout': {
                    'title': 'Distribution of number of labels per message',
                    'yaxis': {
                        'title': "Number of messages"
                    },
                    'xaxis': {
                        'title': "Number of labels"
                    }
                }
            },
            {
                'data': [
                    Bar(
                        x=message_length_distibution.index,
                        y=message_length_distibution.values
                    )
                ],

                'layout': {
                    'title': 'Distribution of message length',
                    'yaxis': {
                        'title': "Number of messages"
                    },
                    'xaxis': {
                        'title': "Length of message"
                    }
                }
            }
        ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()