#!/usr/bin/env python3
"""
Delhi House Price Prediction using Linear Regression

This script performs:
1. Data loading and cleaning
2. Exploratory Data Analysis (EDA)
3. Feature engineering
4. Linear Regression model training
5. Model evaluation

Author: House Price Prediction System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

class DelhiHousePricePredictor:
    """
    A class to predict house prices in Delhi using Linear Regression
    """
    
    def __init__(self, data_path='delhi_house_data.csv'):
        """
        Initialize the predictor with data path
        
        Args:
            data_path (str): Path to the CSV file containing house data
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        
    def load_data(self):
        """Load and display basic information about the dataset"""
        print("=" * 60)
        print("LOADING DELHI HOUSE PRICE DATA")
        print("=" * 60)
        
        self.df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        return self.df
    
    def data_info(self):
        """Display detailed information about the dataset"""
        print("\n" + "=" * 60)
        print("DATA INFORMATION")
        print("=" * 60)
        
        print("\nDataset Info:")
        print(self.df.info())
        
        print("\nMissing Values:")
        missing_values = self.df.isnull().sum()
        print(missing_values[missing_values > 0] if missing_values.any() else "No missing values found!")
        
        print("\nNumerical Statistics:")
        print(self.df.describe())
        
        print("\nCategorical Variables:")
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            print(f"\n{col}:")
            print(self.df[col].value_counts().head())
    
    def clean_data(self):
        """Clean the dataset by handling missing values and outliers"""
        print("\n" + "=" * 60)
        print("DATA CLEANING")
        print("=" * 60)
        
        initial_shape = self.df.shape
        print(f"Initial dataset shape: {initial_shape}")
        
        # Check for missing values
        missing_before = self.df.isnull().sum().sum()
        print(f"Missing values before cleaning: {missing_before}")
        
        # Remove rows with missing values (if any)
        self.df = self.df.dropna()
        
        # Remove unrealistic outliers
        # Remove houses with price > 1000 lakhs (10 crores) or < 5 lakhs
        price_before = len(self.df)
        self.df = self.df[(self.df['Price_lakhs'] >= 5) & (self.df['Price_lakhs'] <= 1000)]
        price_after = len(self.df)
        
        # Remove houses with area < 200 sqft or > 10000 sqft
        area_before = len(self.df)
        self.df = self.df[(self.df['Area_sqft'] >= 200) & (self.df['Area_sqft'] <= 10000)]
        area_after = len(self.df)
        
        # Remove houses with age > 100 years
        age_before = len(self.df)
        self.df = self.df[self.df['Age_years'] <= 100]
        age_after = len(self.df)
        
        final_shape = self.df.shape
        missing_after = self.df.isnull().sum().sum()
        
        print(f"Missing values after cleaning: {missing_after}")
        print(f"Rows removed due to price outliers: {price_before - price_after}")
        print(f"Rows removed due to area outliers: {area_before - area_after}")
        print(f"Rows removed due to age outliers: {age_before - age_after}")
        print(f"Final dataset shape: {final_shape}")
        print(f"Total rows removed: {initial_shape[0] - final_shape[0]}")
        
        return self.df
    
    def exploratory_data_analysis(self):
        """Perform comprehensive EDA with visualizations"""
        print("\n" + "=" * 60)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Price distribution
        plt.subplot(3, 4, 1)
        plt.hist(self.df['Price_lakhs'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of House Prices')
        plt.xlabel('Price (Lakhs)')
        plt.ylabel('Frequency')
        
        # 2. Area vs Price
        plt.subplot(3, 4, 2)
        plt.scatter(self.df['Area_sqft'], self.df['Price_lakhs'], alpha=0.6, color='green')
        plt.title('Area vs Price')
        plt.xlabel('Area (sqft)')
        plt.ylabel('Price (Lakhs)')
        
        # 3. Bedrooms vs Price
        plt.subplot(3, 4, 3)
        bedroom_price = self.df.groupby('Bedrooms')['Price_lakhs'].mean()
        bedroom_price.plot(kind='bar', color='orange')
        plt.title('Average Price by Bedrooms')
        plt.xlabel('Number of Bedrooms')
        plt.ylabel('Average Price (Lakhs)')
        plt.xticks(rotation=0)
        
        # 4. Locality vs Price (top 10)
        plt.subplot(3, 4, 4)
        locality_price = self.df.groupby('Locality')['Price_lakhs'].mean().sort_values(ascending=False).head(10)
        locality_price.plot(kind='barh', color='red')
        plt.title('Top 10 Expensive Localities')
        plt.xlabel('Average Price (Lakhs)')
        
        # 5. Age vs Price
        plt.subplot(3, 4, 5)
        plt.scatter(self.df['Age_years'], self.df['Price_lakhs'], alpha=0.6, color='purple')
        plt.title('Age vs Price')
        plt.xlabel('Building Age (years)')
        plt.ylabel('Price (Lakhs)')
        
        # 6. Furnished status vs Price
        plt.subplot(3, 4, 6)
        furnished_price = self.df.groupby('Furnished_status')['Price_lakhs'].mean()
        furnished_price.plot(kind='bar', color='brown')
        plt.title('Average Price by Furnished Status')
        plt.xlabel('Furnished Status')
        plt.ylabel('Average Price (Lakhs)')
        plt.xticks(rotation=45)
        
        # 7. Metro nearby effect
        plt.subplot(3, 4, 7)
        metro_price = self.df.groupby('Metro_nearby')['Price_lakhs'].mean()
        metro_labels = ['No Metro', 'Metro Nearby']
        plt.bar(metro_labels, metro_price.values, color=['red', 'green'])
        plt.title('Metro Connectivity vs Price')
        plt.ylabel('Average Price (Lakhs)')
        
        # 8. Parking spaces vs Price
        plt.subplot(3, 4, 8)
        parking_price = self.df.groupby('Parking_spaces')['Price_lakhs'].mean()
        parking_price.plot(kind='bar', color='teal')
        plt.title('Average Price by Parking Spaces')
        plt.xlabel('Number of Parking Spaces')
        plt.ylabel('Average Price (Lakhs)')
        plt.xticks(rotation=0)
        
        # 9. Correlation heatmap
        plt.subplot(3, 4, 9)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlation Matrix')
        
        # 10. Box plot of prices by bedrooms
        plt.subplot(3, 4, 10)
        sns.boxplot(data=self.df, x='Bedrooms', y='Price_lakhs')
        plt.title('Price Distribution by Bedrooms')
        plt.xlabel('Number of Bedrooms')
        plt.ylabel('Price (Lakhs)')
        
        # 11. Floor vs Price
        plt.subplot(3, 4, 11)
        floor_price = self.df.groupby('Floor')['Price_lakhs'].mean()
        plt.plot(floor_price.index, floor_price.values, marker='o', color='blue')
        plt.title('Average Price by Floor')
        plt.xlabel('Floor Number')
        plt.ylabel('Average Price (Lakhs)')
        
        # 12. Price trends
        plt.subplot(3, 4, 12)
        price_ranges = pd.cut(self.df['Price_lakhs'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        price_ranges.value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title('Price Range Distribution')
        plt.ylabel('')
        
        plt.tight_layout()
        plt.savefig('delhi_house_eda.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print key insights
        print("\nKEY INSIGHTS:")
        print("-" * 40)
        print(f"1. Average house price: ₹{self.df['Price_lakhs'].mean():.2f} Lakhs")
        print(f"2. Price range: ₹{self.df['Price_lakhs'].min():.2f} - ₹{self.df['Price_lakhs'].max():.2f} Lakhs")
        print(f"3. Most expensive locality: {locality_price.index[0]} (₹{locality_price.iloc[0]:.2f} Lakhs avg)")
        print(f"4. Average area: {self.df['Area_sqft'].mean():.0f} sqft")
        print(f"5. Most common bedroom count: {self.df['Bedrooms'].mode().iloc[0]}")
        print(f"6. Metro connectivity premium: {(metro_price[1] - metro_price[0]):.2f} Lakhs")
        
        # Feature correlations with price
        price_corr = correlation_matrix['Price_lakhs'].sort_values(ascending=False)
        print(f"\n7. Top 5 features correlated with price:")
        for i, (feature, corr) in enumerate(price_corr.head(6)[1:].items(), 1):
            print(f"   {i}. {feature}: {corr:.3f}")
    
    def feature_engineering(self):
        """Prepare features for machine learning"""
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING")
        print("=" * 60)
        
        # Create a copy for feature engineering
        df_features = self.df.copy()
        
        # Create new features
        print("Creating new features...")
        
        # 1. Price per sqft
        df_features['Price_per_sqft'] = df_features['Price_lakhs'] * 100000 / df_features['Area_sqft']
        
        # 2. Room efficiency (area per bedroom)
        df_features['Area_per_bedroom'] = df_features['Area_sqft'] / df_features['Bedrooms']
        
        # 3. Bathroom ratio
        df_features['Bathroom_bedroom_ratio'] = df_features['Bathrooms'] / df_features['Bedrooms']
        
        # 4. High rise indicator (more than 10 floors)
        df_features['Is_highrise'] = (df_features['Total_floors'] > 10).astype(int)
        
        # 5. Floor position (bottom, middle, top)
        def get_floor_position(row):
            floor = row['Floor']
            total = row['Total_floors']
            if floor <= 2:
                return 'Bottom'
            elif floor >= total - 1:
                return 'Top'
            else:
                return 'Middle'
        
        df_features['Floor_position'] = df_features.apply(get_floor_position, axis=1)
        
        # Encode categorical variables
        print("Encoding categorical variables...")
        
        categorical_columns = ['Locality', 'Furnished_status', 'Floor_position']
        
        for col in categorical_columns:
            le = LabelEncoder()
            df_features[f'{col}_encoded'] = le.fit_transform(df_features[col])
            self.label_encoders[col] = le
            print(f"Encoded {col}: {len(le.classes_)} categories")
        
        # Select features for modeling
        feature_columns = [
            'Area_sqft', 'Bedrooms', 'Bathrooms', 'Age_years', 'Parking_spaces',
            'Metro_nearby', 'Floor', 'Total_floors', 'Area_per_bedroom',
            'Bathroom_bedroom_ratio', 'Is_highrise', 'Locality_encoded',
            'Furnished_status_encoded', 'Floor_position_encoded'
        ]
        
        X = df_features[feature_columns]
        y = df_features['Price_lakhs']
        
        print(f"Selected {len(feature_columns)} features for modeling")
        print(f"Feature names: {feature_columns}")
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """Train the Linear Regression model"""
        print("\n" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        
        # Scale the features
        print("\nScaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Train the model
        print("Training Linear Regression model...")
        self.model = LinearRegression()
        self.model.fit(X_train_scaled, self.y_train)
        
        # Make predictions
        train_predictions = self.model.predict(X_train_scaled)
        test_predictions = self.model.predict(X_test_scaled)
        
        print("Model training completed!")
        
        return train_predictions, test_predictions
    
    def evaluate_model(self, train_predictions, test_predictions):
        """Evaluate the model performance"""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        # Training metrics
        train_mae = mean_absolute_error(self.y_train, train_predictions)
        train_mse = mean_squared_error(self.y_train, train_predictions)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(self.y_train, train_predictions)
        
        # Testing metrics
        test_mae = mean_absolute_error(self.y_test, test_predictions)
        test_mse = mean_squared_error(self.y_test, test_predictions)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(self.y_test, test_predictions)
        
        print("TRAINING METRICS:")
        print(f"Mean Absolute Error (MAE): ₹{train_mae:.2f} Lakhs")
        print(f"Root Mean Square Error (RMSE): ₹{train_rmse:.2f} Lakhs")
        print(f"R² Score: {train_r2:.4f}")
        
        print("\nTEST METRICS:")
        print(f"Mean Absolute Error (MAE): ₹{test_mae:.2f} Lakhs")
        print(f"Root Mean Square Error (RMSE): ₹{test_rmse:.2f} Lakhs")
        print(f"R² Score: {test_r2:.4f}")
        
        # Model interpretation
        print(f"\nMODEL INTERPRETATION:")
        print("-" * 30)
        print(f"The model explains {test_r2*100:.1f}% of the variance in house prices")
        print(f"On average, predictions are off by ₹{test_mae:.2f} Lakhs")
        print(f"Typical prediction error: ±₹{test_rmse:.2f} Lakhs")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': self.X_train.columns,
            'Coefficient': self.model.coef_,
            'Abs_Coefficient': np.abs(self.model.coef_)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print(f"\nTOP 10 MOST IMPORTANT FEATURES:")
        print("-" * 40)
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            direction = "increases" if row['Coefficient'] > 0 else "decreases"
            print(f"{i:2d}. {row['Feature']}: {direction} price (coef: {row['Coefficient']:.3f})")
        
        # Visualization
        self.plot_predictions(test_predictions)
        
        return {
            'train_mae': train_mae, 'train_rmse': train_rmse, 'train_r2': train_r2,
            'test_mae': test_mae, 'test_rmse': test_rmse, 'test_r2': test_r2,
            'feature_importance': feature_importance
        }
    
    def plot_predictions(self, test_predictions):
        """Plot prediction results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(self.y_test, test_predictions, alpha=0.6, color='blue')
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Price (Lakhs)')
        axes[0, 0].set_ylabel('Predicted Price (Lakhs)')
        axes[0, 0].set_title('Actual vs Predicted Prices')
        
        # 2. Residual plot
        residuals = self.y_test - test_predictions
        axes[0, 1].scatter(test_predictions, residuals, alpha=0.6, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Price (Lakhs)')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        
        # 3. Residual distribution
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        
        # 4. Feature importance plot
        feature_importance = pd.DataFrame({
            'Feature': self.X_train.columns,
            'Abs_Coefficient': np.abs(self.model.coef_)
        }).sort_values('Abs_Coefficient', ascending=True).tail(10)
        
        axes[1, 1].barh(range(len(feature_importance)), feature_importance['Abs_Coefficient'])
        axes[1, 1].set_yticks(range(len(feature_importance)))
        axes[1, 1].set_yticklabels(feature_importance['Feature'])
        axes[1, 1].set_xlabel('Absolute Coefficient Value')
        axes[1, 1].set_title('Top 10 Feature Importance')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_price(self, house_features):
        """
        Predict price for a new house
        
        Args:
            house_features (dict): Dictionary containing house features
        
        Returns:
            float: Predicted price in Lakhs
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained yet. Please run the complete pipeline first.")
        
        # Create feature vector
        feature_vector = np.array([
            house_features.get('Area_sqft', 1000),
            house_features.get('Bedrooms', 3),
            house_features.get('Bathrooms', 2),
            house_features.get('Age_years', 5),
            house_features.get('Parking_spaces', 1),
            house_features.get('Metro_nearby', 1),
            house_features.get('Floor', 3),
            house_features.get('Total_floors', 10),
            house_features.get('Area_per_bedroom', 333),
            house_features.get('Bathroom_bedroom_ratio', 0.67),
            house_features.get('Is_highrise', 0),
            house_features.get('Locality_encoded', 5),
            house_features.get('Furnished_status_encoded', 1),
            house_features.get('Floor_position_encoded', 1)
        ]).reshape(1, -1)
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Predict
        predicted_price = self.model.predict(feature_vector_scaled)[0]
        
        return predicted_price
    
    def run_complete_pipeline(self):
        """Run the complete house price prediction pipeline"""
        print("🏠 DELHI HOUSE PRICE PREDICTION SYSTEM 🏠")
        print("=" * 60)
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Data information
            self.data_info()
            
            # Step 3: Clean data
            self.clean_data()
            
            # Step 4: EDA
            self.exploratory_data_analysis()
            
            # Step 5: Feature engineering
            X, y = self.feature_engineering()
            
            # Step 6: Train model
            train_pred, test_pred = self.train_model(X, y)
            
            # Step 7: Evaluate model
            metrics = self.evaluate_model(train_pred, test_pred)
            
            print("\n" + "=" * 60)
            print("PIPELINE COMPLETED SUCCESSFULLY! 🎉")
            print("=" * 60)
            
            return metrics
            
        except Exception as e:
            print(f"\nError in pipeline: {str(e)}")
            raise


def main():
    """Main function to demonstrate the Delhi House Price Prediction system"""
    
    # Initialize predictor
    predictor = DelhiHousePricePredictor()
    
    # Run complete pipeline
    metrics = predictor.run_complete_pipeline()
    
    # Example prediction
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTION")
    print("=" * 60)
    
    example_house = {
        'Area_sqft': 1200,
        'Bedrooms': 3,
        'Bathrooms': 2,
        'Age_years': 5,
        'Parking_spaces': 1,
        'Metro_nearby': 1,
        'Floor': 5,
        'Total_floors': 15,
        'Area_per_bedroom': 400,
        'Bathroom_bedroom_ratio': 0.67,
        'Is_highrise': 1,
        'Locality_encoded': 10,  # Mid-range locality
        'Furnished_status_encoded': 1,  # Semi-furnished
        'Floor_position_encoded': 1  # Middle floor
    }
    
    try:
        predicted_price = predictor.predict_price(example_house)
        print(f"Example house features: {example_house}")
        print(f"Predicted price: ₹{predicted_price:.2f} Lakhs")
    except Exception as e:
        print(f"Error in prediction: {str(e)}")


if __name__ == "__main__":
    main()