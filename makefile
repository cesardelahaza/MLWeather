.PHONY: install data models app all

install:
	pip install -r requirements.txt

data:
	python src/historic_data.py
	python src/forecast_data.py
	python src/data_treatment.py

models:
	python models/linear_model.py
	python models/xgboost_model.py
	python models/random_forest_model.py
	python models/sarima_model.py
	python models/lstm_model.py

app:
	streamlit run app.py

all: data models app
