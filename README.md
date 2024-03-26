# Diabetes Prediction App

Streamlit Web App to predict the onset of diabetes based on diagnostic measures.

The data used in this repo is the [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) from Kaggle.
The data contains information on females at least 21 years old of Pima Indian heritage.

Try app [here](https://app-diabetes.streamlit.app)!

## Files

- `app.py`: streamlit app file
- `model.py`: script for generating the Random Forest classifier model
- `diabetes.csv` and `model_weights.mw`: data file and pre-trained model
- `requirements.txt`: package requirements files

## Run App Locally 

### Shell

For directly run streamlit locally in the repo root folder as follows:

```shell
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ streamlit run app.py
```
Open http://localhost:8501 to view the app.



&copy; Bobyleva Ekaterina