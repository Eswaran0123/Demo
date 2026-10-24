# AutoMPG Predictor — Enhanced Edition

## What's new
- Dark mode enabled by default, with a light/dark toggle.
- Animated gradient hero banner, floating glow, hover effects, and metric cards.
- Reset button clears model results, previous prediction, and vehicle input fields.
- Responsive dashboard and styled prediction result.

## Machine-learning features
- Built-in sample dataset or upload a CSV.
- Missing-value imputation, numeric scaling, categorical encoding.
- Compares Linear Regression, Random Forest, and Gradient Boosting.
- Reports R², MAE, and RMSE; predicts MPG and approximate km/L.
- Downloads model comparison as CSV.

## Run on Windows
1. Extract this ZIP.
2. Open Command Prompt in the extracted folder.
3. Install: `py -m pip install -r requirements.txt`
4. Launch: `py -m streamlit run app.py`
5. Open the local URL shown in the terminal, usually `http://localhost:8501`.

The built-in dataset requires internet access unless you put `auto-mpg.csv` in this folder. Python 3.14 package compatibility depends on available builds; if installation fails, try Python 3.12 or 3.13.
