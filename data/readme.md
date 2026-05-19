# Dataset Documentation

## ML-Based Prediction of Restaurant Food Waste for Sustainable Food Management

This directory contains the datasets used to develop a machine learning-based framework for predicting restaurant food waste. The datasets were collected from multiple public sources and combined to construct a structured dataset suitable for predictive analytics and sustainable food management research.

---

# Dataset Structure

```text
data/
│
├── raw/
│   ├── Food Wastage/
│   ├── Restaurant Demand/
│   └── Weather/
│
└── processed/
    └── restaurant_food_waste_final_dataset.csv
```

---

# Dataset Sources

## 1. Restaurant Demand Dataset

This dataset contains restaurant operational and demand-related information used for customer demand forecasting and food preparation analysis.

### Features
- Restaurant ID
- Meal ID
- Number of Orders
- Checkout Price
- Base Price
- Promotional Features
- Operational Area
- Meal Categories

### Source
https://www.kaggle.com/datasets/kannanaikkal/food-demand-forecasting

---

## 2. Global Food Wastage Dataset

This dataset provides food wastage and sustainability-related information that supports food waste estimation and environmental analysis.

### Features
- Food Waste Quantity
- Food Category
- Economic Loss
- Disposal Information
- Environmental Impact

### Source
https://www.kaggle.com/datasets/joebeachcapital/food-waste

---

## 3. Weather and Contextual Dataset

This dataset contains environmental and contextual information used to improve food demand and food waste prediction performance.

### Features
- Temperature
- Rainfall
- Humidity
- Weather Conditions
- Seasonal Information

### Source
https://www.kaggle.com/datasets/selfishgene/historical-hourly-weather-data

---

# Final Processed Dataset

## File
`restaurant_food_waste_final_dataset.csv`

The processed dataset was created by integrating restaurant operational data, food wastage information, and weather-contextual variables.

---

# Dataset Merging Strategy

The datasets were merged using common temporal and contextual attributes such as:

- Date
- Restaurant Identifier
- Regional Information
- Time-based Features

Additional feature engineering techniques were applied to generate:

- Year
- Month
- Week
- Day of Week
- Weekend Indicator
- Holiday Indicator
- Estimated Customers
- Food Prepared Quantity
- Food Sold Quantity
- Food Waste Quantity

---

# Final Dataset Features

| Feature | Description |
|---|---|
| date | Daily record date |
| restaurant_id | Unique restaurant identifier |
| city_code | City identifier |
| region_code | Regional identifier |
| center_type | Restaurant type |
| op_area | Operational area |
| unique_meals | Number of meal types |
| dominant_category | Main food category |
| dominant_cuisine | Main cuisine type |
| avg_checkout_price | Average checkout price |
| avg_base_price | Average base price |
| emailer_promo_rate | Promotional email rate |
| homepage_feature_rate | Homepage promotion rate |
| is_weekend | Weekend indicator |
| is_holiday | Holiday indicator |
| special_event | Event indicator |
| temperature_c | Temperature in Celsius |
| precipitation_mm | Rainfall information |
| estimated_customers | Estimated customer count |
| num_orders | Total food orders |
| food_sold_kg | Quantity of food sold |
| food_prepared_kg | Quantity of food prepared |
| food_waste_kg | Quantity of food wasted |

---

# Research Objective

The primary objective of this dataset is to support machine learning research for:

- Restaurant food waste prediction
- Demand forecasting
- Sustainable restaurant management
- Resource optimization
- Operational decision support

---

# Machine Learning Applications

The dataset can be used for:

- Regression Analysis
- Food Waste Prediction
- Demand Forecasting
- Sustainability Analytics
- Restaurant Decision Support Systems

---

# Recommended Models

The following machine learning models were evaluated in this project:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

Among them, Random Forest achieved the best predictive performance for restaurant food waste estimation.

---

# Citation

If you use this dataset or repository in your research, please cite:

```bibtex
@misc{naeem2026foodwaste,
  title={ML-Based Prediction of Restaurant Food Waste for Sustainable Food Management},
  author={Mehedi Naeem},
  year={2026},
  publisher={GitHub},
  url={https://github.com/mehedinaeem/ml-based-prediction-of-restaurant-food-waste-for-sustainable-food-management}
}
```

---

# License

This dataset and repository are distributed for academic and research purposes.

Please check the original dataset licenses from their respective sources before commercial use.

---

# Author

Mehedi Naeem

Department of Computer Science and Engineering

Research Area:
- Machine Learning
- Predictive Analytics
- Sustainable Computing
- Food Waste Management