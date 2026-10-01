# Car-price-prediction-with-machine-learning

## Objective
Predict car prices using available vehicle features.

## Algorithm
Random Forest Regression with preprocessing for numeric and categorical features.

## Dataset
Use the car-price dataset supplied by CodeAlpha. Put it in this folder as `car_price.csv`, or change `FILE` in the Python file.

## Important
Set `TARGET` to the exact price column in the dataset before running.

## Run
```bash
pip install -r requirements.txt
python car_price_prediction.py
```

## Evaluation
The script reports MAE, RMSE and R² and creates `actual_vs_predicted.png`.
