# ML-Based Prediction of Restaurant Food Waste for Sustainable Food Management

## Overview

This repository contains a comprehensive dataset and analysis for predicting restaurant food waste using machine learning techniques. The dataset includes operational, financial, and environmental data from multiple restaurants across different regions, with the goal of helping restaurants reduce food waste and improve sustainability practices.

## Dataset Description

### Dataset File
- **Name:** `restaurant_food_waste_final_dataset.csv`
- **Time Period:** 2018 onwards
- **Format:** CSV (Comma-Separated Values)
- **Records:** Multiple restaurants with daily operational metrics

### Key Features

The dataset contains 19 distinct features organized into the following categories:

#### Temporal Features
- `date`: Transaction/operational date
- `year`: Year of the record
- `month`: Month (1-12)
- `week`: Week number
- `day_of_week`: Day of the week (e.g., Monday, Tuesday)
- `is_weekend`: Binary indicator (1 if weekend, 0 otherwise)
- `is_holiday`: Binary indicator (1 if holiday, 0 otherwise)
- `special_event`: Binary indicator for special events

#### Restaurant/Location Features
- `restaurant_id`: Unique identifier for the restaurant
- `city_code`: Geographic city code
- `region_code`: Geographic region code
- `center_type`: Restaurant type classification (TYPE_A, TYPE_B, TYPE_C)
- `op_area`: Operating area of the restaurant (in square units)

#### Menu & Category Features
- `unique_meals`: Number of unique meals offered
- `dominant_category`: Primary food category (e.g., Beverages, Sandwich)
- `dominant_cuisine`: Primary cuisine type (e.g., Thai, Italian, Continental)

#### Pricing Features
- `avg_checkout_price`: Average price at checkout
- `avg_base_price`: Average base price

#### Marketing/Promotion Features
- `emailer_promo_rate`: Rate of email promotions (0-1)
- `homepage_feature_rate`: Rate of homepage feature promotions (0-1)

#### Environmental Features
- `temperature_c`: Average temperature in Celsius
- `precipitation_mm`: Average precipitation in millimeters

#### Operational/Sales Features
- `estimated_customers`: Estimated number of customers
- `num_orders`: Number of orders placed
- `food_sold_kg`: Amount of food sold in kilograms
- `food_prepared_kg`: Amount of food prepared in kilograms

#### Target Variable
- `food_waste_kg`: **Amount of food wasted in kilograms** (Primary prediction target)

## Problem Statement

Food waste is a critical global issue affecting sustainability, profitability, and environmental impact. Restaurants typically overprepare food to meet demand fluctuations, leading to significant waste. This dataset enables:

1. **Predictive Modeling:** Build ML models to predict daily food waste based on operational and environmental factors
2. **Pattern Recognition:** Identify key drivers of food waste in different restaurant types
3. **Decision Support:** Help restaurants optimize food preparation and reduce waste
4. **Sustainability:** Contribute to sustainable food management practices

## Data Characteristics

### Statistical Summary
- **Multiple restaurant types** (TYPE_A, TYPE_B, TYPE_C) with varying operational scales
- **Daily granularity** allowing for time-series analysis
- **Geographic diversity** across multiple cities and regions
- **Weather correlation** with operational metrics
- **Promotional activities** tracking email and homepage campaigns

### Data Quality
- Clean, structured format suitable for ML applications
- Balanced feature set covering operational, environmental, and market factors
- Spanning multiple years of business operations

## Potential Applications

### Machine Learning Tasks
1. **Regression:** Predict continuous food waste quantities (food_waste_kg)
2. **Time Series Forecasting:** Predict future food waste based on historical patterns
3. **Classification:** Categorize restaurants by waste severity (high/medium/low)
4. **Clustering:** Identify restaurant groups with similar waste patterns

### Business Insights
- Identify which restaurant types generate more waste
- Analyze impact of weather on food waste
- Evaluate effectiveness of promotional strategies on waste reduction
- Seasonal trend analysis

## Column Definitions Summary

| Column | Type | Description |
|--------|------|-------------|
| date | Date | Operational date |
| year, month, week, day_of_week | Integer/String | Temporal breakdown |
| is_weekend | Binary | 1 if weekend, 0 otherwise |
| is_holiday | Binary | 1 if public holiday, 0 otherwise |
| special_event | Binary | 1 if special event occurred, 0 otherwise |
| restaurant_id | Integer | Unique restaurant identifier |
| city_code | Integer | Geographic city code |
| region_code | Integer | Geographic region code |
| center_type | String | Restaurant classification (TYPE_A/B/C) |
| op_area | Float | Operating area (sq units) |
| unique_meals | Integer | Number of meal options |
| dominant_category | String | Primary food category |
| dominant_cuisine | String | Primary cuisine type |
| avg_checkout_price | Float | Average customer checkout amount |
| avg_base_price | Float | Average menu base price |
| emailer_promo_rate | Float | Email promotion frequency (0-1) |
| homepage_feature_rate | Float | Homepage feature frequency (0-1) |
| temperature_c | Float | Average temperature (°C) |
| precipitation_mm | Float | Average precipitation (mm) |
| estimated_customers | Integer | Expected customer count |
| num_orders | Integer | Total orders placed |
| food_sold_kg | Float | Food sold (kg) |
| food_prepared_kg | Float | Food prepared (kg) |
| **food_waste_kg** | **Float** | **Food wasted (kg) - TARGET VARIABLE** |

## Usage

This dataset is suitable for:
- Machine learning model development and training
- Exploratory data analysis (EDA)
- Feature engineering and selection
- Model validation and benchmarking
- Academic research on food waste prediction
- Business intelligence and analytics

## Sustainability Impact

Accurate food waste prediction can help restaurants:
- ✅ Reduce operational costs through better inventory management
- ✅ Minimize environmental impact and carbon footprint
- ✅ Improve food supply chain efficiency
- ✅ Support sustainable business practices
- ✅ Contribute to global food security initiatives

## File Structure

```
restaurant-food-waste-for-sustainable-food-management/
├── README.md (this file)
├── restaurant_food_waste_final_dataset.csv
└── [Additional analysis and model files]
```

## Notes

- The data spans from 2018 onwards with daily granularity
- Multiple restaurants across different geographic locations and types are represented
- Environmental factors (temperature, precipitation) are included to capture seasonal impacts
- Promotional activities are tracked to evaluate their correlation with waste levels

## Contributing

For improvements or corrections to this dataset documentation, please create an issue or pull request.

## License

This project is licensed under the MIT License.

## Citation

If you use this dataset in your research or project, please cite our work:

> Naeem, Md Mehedi Hasan, and Md Ashraful Islam. "Machine Learning-Based Prediction of Restaurant Food Waste for Sustainable Food Management." *Jatiya Kabi Kazi Nazrul Islam University, Bangladesh*, 2026.

**BibTeX:**
```bibtex
@article{naeem2026foodwaste,
  title       = {Machine Learning-Based Prediction of Restaurant Food Waste for Sustainable Food Management},
  author      = {Naeem, Md Mehedi Hasan and Islam, Md Ashraful},
  year        = {2026},
  institution = {Jatiya Kabi Kazi Nazrul Islam University, Bangladesh}
}
```

---

**Last Updated:** March 2026
**Dataset Version:** Final
