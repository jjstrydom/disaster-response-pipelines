# disaster-response-pipelines
A project to showcase the use of github, ETL, ML, and deployment using python.

## Environment
The analysis uses python. The environment is managed using `pipenv`. [This link](https://realpython.com/pipenv-guide/) will help you get started with `pipenv`.

This project uses `python 3.7` and the following packages:
- pandas
- sqlalchemy
- nltk
- sklearn
- flask

## Dataset
Dataset is built from `disaster_categories.csv` and `disaster_messages.csv` included in the repo in the `data` folder. 

## Files
- `app/run.py`: Main program to run the dashboard.
- `data/process_data.py`: Script that performs ETL on the data.
- `models/train_classifier.py`: Script that performs model training.
- `LICENSE`: MIT. Read for information on re-use and sharing. Usage of this software, analysis or anything else in this repository is subject to the license.
- `README.md`: This file.
