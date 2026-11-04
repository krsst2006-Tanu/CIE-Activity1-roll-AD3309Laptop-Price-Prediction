# Laptop Price Prediction using Linear Regression

## Project objective
This beginner-level Machine Learning project predicts an estimated laptop price from its specifications using **Multiple Linear Regression**.

## Input features
- RAM (GB)
- Storage (GB)
- Processor score (a simple educational rating)
- Screen size (inches)
- Weight (kg)

## Target
- Laptop price in Indian rupees (INR)

## Tools used
Python, Pandas, Matplotlib, NumPy and Scikit-learn.

## Dataset note
`laptop_data.csv` is a small **synthetic educational dataset** created for learning and demonstrating the project. It is not collected from retailers and must not be treated as current or reliable market pricing data.

## How to run
1. Download or clone this repository.
2. Install the libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Keep `laptop_price_prediction.py` and `laptop_data.csv` in the same folder.
4. Run:
   ```bash
   python laptop_price_prediction.py
   ```

You can also upload the CSV and Python file to Google Colab and run the code there.

## Project workflow
1. Load the dataset.
2. Select laptop specifications as input features and price as the target.
3. Split the data into training and testing sets.
4. Train a Linear Regression model.
5. Predict prices for test data.
6. Evaluate the model using MAE, RMSE and R² score.
7. Plot actual prices against predicted prices.

## Expected output
The program prints evaluation measures and an estimated price for a sample laptop. It also displays an actual-versus-predicted scatter plot. Exact results are produced when the code is run.

## Learning outcome
Understand how a regression model learns the relationship between input features and a continuous value (laptop price), and how to make and evaluate predictions.
