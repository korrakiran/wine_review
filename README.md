# Wine Quality Prediction

This project is a machine learning approach to predict wine scores (`points`) based on structured features such as country, province, variety, price, and winery. The goal is to demonstrate regression techniques using classical ML algorithms on a real-world dataset.

## Dataset

The dataset used is the [Wine Reviews Dataset](https://github.com/korrakiran/wine_review/blob/main/wine_preprocessed.csv), which contains over 130,000 wine reviews with the following relevant columns:

- `country`, `province`, `region_1`, `region_2` (categorical)
- `variety`, `winery` (categorical)
- `price` (numeric)
- `points` (numeric, target variable)

Irrelevant or text-heavy columns like `description`, `title`, `taster_name`, and `taster_twitter_handle` were dropped for this regression task.

## Preprocessing

- Missing values in numeric features (`price`) were filled with the median.  
- Categorical columns were encoded to numeric values using `LabelEncoder`.  
- Rows with missing target values (`points`) were removed.  

## Models Used

The following regression models were evaluated:

1. **Random Forest Regressor**  
2. **XGBoost Regressor**  
3. **Linear Regression**

## Evaluation Metrics

The models were evaluated using:

- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R² Score**

### Results

| Model            | MSE     | MAE    | R²      |
|-----------------|---------|--------|---------|
| Random Forest    | 4.70    | 1.68   | 0.498   |
| XGBoost          | 4.81    | 1.73   | –       |
| Linear Regression| 10.36   | 2.56   | -0.105  |

**Conclusion:**  
Random Forest performed the best overall, achieving the lowest error and explaining ~50% of the variance in wine scores. Linear Regression underperformed, highlighting the importance of using models capable of handling non-linear relationships and complex feature interactions.

## Future Work

- Incorporate textual features from the `description` column using NLP techniques to improve prediction accuracy.  
- Hyperparameter tuning for Random Forest and XGBoost to further reduce error.  
- Explore feature engineering with price, regions, and winery combinations.

## How to Run

1. [Clone the repository](https://github.com/korrakiran/wine_review)
2. Install the required libraries:  

```bash
pip install pandas scikit-learn xgboost
