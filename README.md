# disaster-response-pipelines

Using labelled message data from real life disasters to aid in disaster response by classifying unseen messages to the corresponding disaster response team.
Typically there are multiple independent teams each tackling a different task of a disaster response.
There are thousands of messages sent during a disaster, with some containing critical information relevant to a specific disaster response team. 
To manually filter and assign these messages would be too intensive, time consuming, and slow for the overall response effort.
The disaster response piple aims to take the messages and automatically assign them to the correct disaster response team.

A project to showcase the use of github, ETL, ML, and deployment using python.

## Environment
The analysis uses python. The environment is managed using `pipenv`. [This link](https://realpython.com/pipenv-guide/) will help you get started with `pipenv`.

This project uses `python 3.7` and the following packages:
- pandas
- sqlalchemy
- nltk
- sklearn
- flask
- joblib
- plotly
- psutil

For the most up to date information see Pipfile in the root folder.

## Dataset
Dataset is built from `disaster_categories.csv` and `disaster_messages.csv` included in the repo in the `data` folder. 

## Files
- `app/run.py`: Main program to run the dashboard.
- `data/process_data.py`: Script that performs ETL on the data.
- `models/train_classifier.py`: Script that performs model training.
- `LICENSE`: MIT. Read for information on re-use and sharing. Usage of this software, analysis or anything else in this repository is subject to the license.
- `README.md`: This file.

## How to run
### Process data
Navigate to the `/data` folder and execute
`python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`
### Train classifier
Navigate to the `/models` folder and execute
`python train_classifier.py ../data/DisasterResponse.db classifier.pkl`
### View dashboard
Navigate to the `/app` folder and execute
`python run.py`. 
In the terminal the script will print the host address of the dashboard 
(typically [`http://0.0.0.0:3001/`](http://0.0.0.0:3001/) when running locally). 
Open this address in your browser to view the dashboard.

# Model Performance

| class                  | precision | recall    | f1score   | accuracy  | AUC       |
| -----------------------|-----------|-----------|-----------|-----------|-----------|
| related                |  0.816    | 0.965     | 0.884     | 0.809     |  0.643    |
| request                |  0.859    | 0.494     | 0.628     | 0.899     |  0.739    |
| offer                  |  0.000    | 0.000     | 0.000     | 0.998     |  0.500    |
| aid_related            |  0.767    | 0.700     | 0.732     | 0.789     |  0.776    |
| medical_help           |  0.528    | 0.045     | 0.084     | 0.920     |  0.521    |
| medical_products       |  0.852    | 0.082     | 0.150     | 0.950     |  0.541    |
| search_and_rescue      |  0.556    | 0.120     | 0.197     | 0.977     |  0.559    |
| security               |  0.000    | 0.000     | 0.000     | 0.983     |  0.500    |
| military               |  0.526    | 0.058     | 0.105     | 0.968     |  0.528    |
| child_alone            |  0.000    | 0.000     | 0.000     | 1.000     |    nan    |
| water                  |  0.910    | 0.357     | 0.513     | 0.956     |  0.677    |
| food                   |  0.815    | 0.560     | 0.664     | 0.935     |  0.772    |
| shelter                |  0.860    | 0.331     | 0.478     | 0.936     |  0.663    |
| clothing               |  0.750    | 0.077     | 0.140     | 0.986     |  0.538    |
| money                  |  1.000    | 0.026     | 0.050     | 0.978     |  0.513    |
| missing_people         |  1.000    | 0.016     | 0.032     | 0.989     |  0.508    |
| refugees               |  0.667    | 0.033     | 0.063     | 0.966     |  0.516    |
| death                  |  0.824    | 0.130     | 0.224     | 0.963     |  0.564    |
| other_aid              |  0.818    | 0.027     | 0.052     | 0.874     |  0.513    |
| infrastructure_related |  0.250    | 0.003     | 0.006     | 0.934     |  0.501    |
| transport              |  0.743    | 0.112     | 0.195     | 0.959     |  0.555    |
| buildings              |  0.717    | 0.129     | 0.219     | 0.955     |  0.563    |
| electricity            |  0.600    | 0.026     | 0.049     | 0.978     |  0.513    |
| tools                  |  0.000    | 0.000     | 0.000     | 0.995     |  0.500    |
| hospitals              |  0.000    | 0.000     | 0.000     | 0.988     |  0.500    |
| shops                  |  0.000    | 0.000     | 0.000     | 0.996     |  0.500    |
| aid_centers            |  0.000    | 0.000     | 0.000     | 0.990     |  0.500    |
| other_infrastructure   |  0.000    | 0.000     | 0.000     | 0.954     |  0.500    |
| weather_related        |  0.851    | 0.686     | 0.759     | 0.883     |  0.821    |
| floods                 |  0.953    | 0.459     | 0.619     | 0.953     |  0.728    |
| storm                  |  0.753    | 0.503     | 0.604     | 0.946     |  0.744    |
| fire                   |  0.000    | 0.000     | 0.000     | 0.988     |  0.500    |
| earthquake             |  0.896    | 0.799     | 0.845     | 0.973     |  0.895    |
| cold                   |  0.875    | 0.068     | 0.126     | 0.982     |  0.534    |
| other_weather          |  0.556    | 0.019     | 0.036     | 0.949     |  0.509    |
| direct_report          |  0.808    | 0.355     | 0.493     | 0.856     |  0.667    |
| **OVERALL**            | **0.762** | **0.531** | **0.568** | **0.256** | **0.589** |

# Development methodology
Project development takes place on the `develop` branch. Stable versions are merged into the `main` branch from the `develop` branch. 