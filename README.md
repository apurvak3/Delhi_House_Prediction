# Delhi House Price Prediction 🏠

A comprehensive machine learning project for predicting house prices in Delhi using Linear Regression. This project demonstrates the complete data science pipeline from data generation to model deployment.

## 🎯 Project Overview

This project implements a house price prediction system for Delhi that includes:
- **Data Generation**: Synthetic Delhi house price dataset with realistic features
- **Data Cleaning**: Handling outliers and data preprocessing
- **Exploratory Data Analysis (EDA)**: Comprehensive visualizations and insights
- **Feature Engineering**: Creating new meaningful features
- **Model Training**: Linear Regression implementation using Scikit-Learn
- **Model Evaluation**: Performance metrics and validation
- **Interactive Predictions**: Easy-to-use prediction interface

## 📊 Dataset Features

The dataset includes the following features:
- **Area_sqft**: House area in square feet
- **Bedrooms**: Number of bedrooms
- **Bathrooms**: Number of bathrooms  
- **Age_years**: Building age in years
- **Locality**: Delhi locality/area (20 different areas)
- **Parking_spaces**: Number of parking spaces
- **Furnished_status**: Unfurnished/Semi-Furnished/Furnished
- **Metro_nearby**: Metro connectivity (1 if within 1km, 0 otherwise)
- **Floor**: Floor number
- **Total_floors**: Total floors in the building
- **Price_lakhs**: House price in Indian Lakhs (Target variable)

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Project

1. **Command Line Interface**:
```bash
python delhi_house_prediction.py
```

2. **Jupyter Notebook** (Interactive):
```bash
jupyter notebook delhi_house_prediction.ipynb
```

## 📁 Project Structure

```
Delhi_House_Prediction/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── delhi_house_data.csv              # Generated dataset
├── delhi_house_prediction.py         # Main Python script
├── delhi_house_prediction.ipynb      # Jupyter notebook
├── delhi_house_eda.png              # EDA visualization
└── model_evaluation.png             # Model evaluation plots
```

## 🔍 Key Features

### 1. Data Generation
- Realistic synthetic dataset with 1000+ house records
- Delhi-specific localities with appropriate price multipliers
- Multiple features affecting house prices

### 2. Data Analysis
- **Price Distribution**: Understanding price ranges across Delhi
- **Locality Analysis**: Most and least expensive areas
- **Feature Correlations**: Relationship between features and prices
- **Metro Impact**: Premium for metro connectivity

### 3. Machine Learning Model
- **Algorithm**: Linear Regression
- **Performance**: ~40% R² score
- **Features**: 14 engineered features
- **Validation**: Train-test split with comprehensive metrics

### 4. Visualizations
- Price distribution histograms
- Correlation heatmaps
- Locality-wise price comparisons
- Model performance plots
- Feature importance analysis

## 📈 Model Performance

- **R² Score**: 0.407 (explains 40.7% of price variance)
- **Mean Absolute Error**: ₹64.21 Lakhs
- **Root Mean Square Error**: ₹84.54 Lakhs

### Top 5 Important Features:
1. **Bedrooms**: Increases price significantly
2. **Bathrooms**: Complex relationship with price
3. **Area_sqft**: Strong positive correlation with price
4. **Age_years**: Newer houses cost more
5. **Locality**: Location is crucial for pricing

## 🏠 Example Predictions

```python
# Example house in Khan Market
house = {
    "area_sqft": 1200,
    "bedrooms": 3,
    "bathrooms": 2,
    "locality": "Khan Market",
    "age_years": 5,
    "metro_nearby": 1
}
# Predicted Price: ₹180-200 Lakhs (approx)
```

## 📊 Key Insights

1. **Location Premium**: Connaught Place commands highest prices (₹280+ Lakhs avg)
2. **Metro Effect**: Metro connectivity adds ~₹13-15 Lakhs premium
3. **Size Matters**: Strong correlation between area and price (0.458)
4. **Age Factor**: Newer houses preferred, prices decrease ~2% per year
5. **Sweet Spot**: 3-bedroom houses are most common and well-priced

## 🛠️ Technical Implementation

### Libraries Used:
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib/seaborn**: Data visualization
- **jupyter**: Interactive development

### Model Pipeline:
1. **Data Loading** → Load CSV dataset
2. **Data Cleaning** → Remove outliers and handle missing values
3. **EDA** → Explore patterns and relationships
4. **Feature Engineering** → Create new meaningful features
5. **Model Training** → Train Linear Regression model
6. **Evaluation** → Assess model performance
7. **Prediction** → Generate price predictions

## 🎯 Future Enhancements

1. **Advanced Models**: Random Forest, Gradient Boosting, Neural Networks
2. **More Features**: Amenities, crime rates, school ratings, traffic data
3. **Real Data**: Integration with actual property listings
4. **Web Interface**: Flask/Django web application
5. **API Development**: REST API for price predictions
6. **Time Series**: Price trend analysis over time

## 📝 Usage Examples

### Basic Prediction
```python
from delhi_house_prediction import DelhiHousePricePredictor

predictor = DelhiHousePricePredictor()
predictor.run_complete_pipeline()

# Predict for a new house
price = predictor.predict_price({
    'Area_sqft': 1500,
    'Bedrooms': 3,
    'Locality_encoded': 10
})
print(f"Predicted Price: ₹{price:.2f} Lakhs")
```

### Custom Analysis
```python
# Load and analyze specific localities
df = pd.read_csv('delhi_house_data.csv')
expensive_areas = df.groupby('Locality')['Price_lakhs'].mean().nlargest(5)
print("Top 5 expensive localities:")
print(expensive_areas)
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Delhi House Price Prediction System**
- Comprehensive ML pipeline for real estate price prediction
- Educational project demonstrating data science best practices

## 🙏 Acknowledgments

- Scikit-Learn for machine learning algorithms
- Pandas and NumPy for data processing
- Matplotlib and Seaborn for visualizations
- Jupyter for interactive development environment

---

*Built with ❤️ for the Delhi real estate market*
