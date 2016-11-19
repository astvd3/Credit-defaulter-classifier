# Credit-defaulter-classifier (Capstone Project)
## Finding out potential credit card defaulters by profiling customers

The goal of this project is to create a model that can be used to classify a set of credit card customers into safe and potential defaulter set by considering some attributes about themselves and their trends in billing and payments.

### Install

This project requires **Python 2.7** with the following library installed:
- [pandas](http://pandas.pydata.org/)
- [numpy](http://www.numpy.org/)
- [sklearn](http://scikit-learn.org/stable/install.html)

### Data

The datasets can be found in the same directory:
`train.csv`
`test.csv`

They can also be downloaded from the UCI Repository (https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).

Note: [Pickle files](https://docs.python.org/2/library/pickle.html) are also present in this folder, which holds the 'best' classifier which was obtained in the Testing phase. This pickle file is further loaded into the app.py file where it is used to create a GUI based application.

### Run

For obtaining the classifier, in a terminal or command window, run the following command:

`python main.py`

This will run the `main.py` file and execute the machine learning code.

For opening the GUI, run the following command

`python app.py`

This will open the GUI where you can enter the attributes and the pop-up will show if the user is a potential defaulter or is safe.
