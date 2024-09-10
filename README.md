# Stock-Price-Prediction-Using-LSTM

This project focuses on building a machine learning model to predict stock prices. The dataset used for training and testing the model consists of historical stock prices obtained from a reliable financial data source.

## Project Overview

The project involves the following steps:

1. **Importing the Dataset**: Historical stock price data is imported using pandas.

2. **Data Cleaning and Preprocessing**: The dataset is cleaned and preprocessed to ensure that it is suitable for model training. This includes handling missing values, converting date columns to datetime objects, and normalizing numerical features.

3. **Feature Engineering**: Additional features are created to enhance the predictive power of the model. This includes creating lag features, rolling averages, and technical indicators.

4. **Train-Test Split**: The dataset is split into training and testing sets to evaluate the performance of the trained model.

5. **Training Models Using Various Regressors**: Multiple regression models such as Linear Regression, Decision Tree, Random Forest, and LSTM (Long Short-Term Memory) networks are trained on the training data to predict future stock prices.

6. **Model Evaluation**: The performance of the trained models is evaluated using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared score.

## Code Overview

The project code consists of the following main components:

- **Importing the Dataset**: Pandas is used to read the dataset file and load it into a DataFrame.

- **Data Cleaning and Preprocessing**: Techniques such as handling missing values, normalizing data, and converting date columns are employed to preprocess the dataset.

- **Feature Engineering**: New features such as lag features, rolling averages, and technical indicators are created to improve model performance.

- **Train-Test Split**: The dataset is split into training and testing sets using the train_test_split function from scikit-learn.

- **Training Models Using Various Regressors**: Multiple regression models are trained on the training data. This includes using scikit-learn for traditional models and Keras for deep learning models like LSTM.

- **Model Evaluation**: The performance of the trained models is evaluated using metrics such as MAE, MSE, and R-squared score.

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- keras
- matplotlib

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the main script to train and evaluate the models:

```bash
python main.py
```


## Acknowledgments

- This project uses historical stock price data.
- Special thanks to the open-source community for providing valuable libraries and resources.

Feel free to explore the code, modify it, and contribute to the project. Happy predicting!
