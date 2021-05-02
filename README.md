# Classify messages sent as a part of disaster response
Repository containing code for a pipeline that classifies messages sent during a disaster response

## Installation

The code contained in this repository was written in HTML and Python 3, and requires the following Python packages: json, plotly, pandas, nltk, flask, sklearn, sqlalchemy, sys, numpy, re, os, pickle.

## Project Overview
This repository contains code for a web app which an emergency worker could use during a disaster event (e.g. an earthquake or hurricane), to classify a disaster message into several categories, in order that the message can be directed to the appropriate aid agencies.

The app uses a ML model to categorize any new messages received, and the repository also contains the code used to train the model and to prepare any new datasets for model training purposes.

## File Descriptions

*process_data.py*: This code takes as its input csv files containing message data and message categories (labels), and creates an SQLite database containing a merged and cleaned version of this data.
*train_classifier.py*: This code takes the SQLite database produced by process_data.py as an input and uses the data contained within it to train and tune a ML model for categorizing messages. The output is a pickle file containing the fitted model. Test evaluation metrics are also printed as part of the training process.
*ETL Pipeline Preparation.ipynb*: The code and analysis contained in this Jupyter notebook was used in the development of process_data.py. process_data.py effectively automates this notebook.
*ML Pipeline Preparation.ipynb*: The code and analysis contained in this Jupyter notebook was used in the development of train_classifier.py. In particular, it contains the analysis used to tune the ML model and determine which algorithm to use. train_classifier.py effectively automates the model fitting process contained in this notebook.
*data*: This folder contains sample messages and categories datasets in csv format.
*app*: This folder contains all of the files necessary to run and render the web app.

## Running Instructions

#### Run *process_data.py*
Save the data folder in the current working directory and process_data.py in the data folder.
From the current working directory, run the following command: python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

#### Run *train_classifier.py*
In the current working directory, create a folder called 'models' and save train_classifier.py in this.
From the current working directory, run the following command: python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

#### Run the web app using *run.py*
Save the app folder in the current working directory.
Run the following command in the app directory: python run.py
Go to http://0.0.0.0:3001/

### Screenshots

![visualization1](https://user-images.githubusercontent.com/10462415/116804179-9636e980-aad1-11eb-893e-4c07819ab95b.png)

![visualization2](https://user-images.githubusercontent.com/10462415/116804271-5d4b4480-aad2-11eb-95a5-3b5874c86df1.png)

![Screen Shot 2021-05-01 at 11 11 43 PM](https://user-images.githubusercontent.com/10462415/116804344-db0f5000-aad2-11eb-8f86-d7bc7c4dd90c.png)

### Acknowledgements

- Udacity for providing an amazing Data Science Nanodegree Program
- Figure Eight for providing the relevant dataset to train the model
