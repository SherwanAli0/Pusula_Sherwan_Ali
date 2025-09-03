# Pusula_Sherwan_Ali
# Physical Medicine & Rehabilitation Dataset Analysis

**Name:** Sherwan Ali  
**Surname:** Ali  
**Email:** Sheroo16.ali@gmail.com

## Project Overview

This project analyzes a medical rehabilitation dataset containing 2,235 patient treatment records. The main goal is to explore the data and prepare it for machine learning models that could predict treatment duration (TedaviSuresi).

## Dataset Description

The dataset contains information about patients receiving physical medicine and rehabilitation treatments:

- **Records:** 2,235 treatment sessions
- **Features:** 13 columns including patient demographics, medical conditions, and treatment details
- **Target Variable:** TedaviSuresi (Treatment Duration in sessions)

### Column Details

| Column | Description |
|--------|-------------|
| HastaNo | Patient ID |
| Yas | Patient Age |
| Cinsiyet | Gender |
| KanGrubu | Blood Type |
| Uyruk | Nationality |
| KronikHastalik | Chronic Diseases |
| Bolum | Department |
| Alerji | Allergies |
| Tanilar | Diagnoses |
| TedaviAdi | Treatment Name |
| TedaviSuresi | Treatment Duration (target variable) |
| UygulamaYerleri | Application Areas |
| UygulamaSuresi | Application Duration |

## Analysis Results

### Key Findings

- Treatment duration ranges from 1 to 37 sessions
- Most patients (75%) receive exactly 15 sessions
- Average treatment duration is 14.6 sessions
- Missing data is highest in allergy information (42%) and blood type (30%)
- Weak positive correlation between patient age and treatment duration

### Data Quality Issues

The dataset had several challenges that were addressed:

- High missing values in clinical fields (allergies, blood type, chronic diseases)
- Text-based treatment duration needed conversion to numeric format
- Comma-separated values in chronic diseases and allergies fields
- Multiple entries per patient representing different treatment episodes

## Data Preprocessing

The preprocessing pipeline includes several steps to clean and prepare the data:

### 1. Missing Value Handling
- Demographic fields: Filled with most common values
- Clinical fields with high missing rates: Marked as "Unknown" 
- Text fields: Used domain-specific defaults like "No known allergies"

### 2. Feature Engineering
- Extracted numeric values from treatment duration text
- Created age categories (Young, Adult, Middle Age, Senior)
- Added binary indicators for chronic disease and allergy presence
- Split comma-separated medical conditions into individual features

### 3. Data Encoding
- One-hot encoded categorical variables (gender, blood type, nationality)
- Created binary features for top chronic diseases and allergies
- Standardized numerical features (age) for modeling

## Technical Implementation

### Libraries Used
- pandas: Data manipulation and analysis
- numpy: Numerical operations
- matplotlib/seaborn: Data visualization
- scikit-learn: Preprocessing tools

### Code Structure
The analysis is organized into modular functions:
- `handle_missing_values()`: Fills missing data using appropriate strategies
- `feature_engineering()`: Creates new features from existing data
- `create_binary_features()`: Processes comma-separated text fields
- `encode_and_scale()`: Handles categorical encoding and numerical scaling
- `preprocess_data()`: Main pipeline combining all preprocessing steps

## Files Generated

- `processed_medical_data_complete.csv`: Complete processed dataset
- `model_features.csv`: Features ready for machine learning
- `model_target.csv`: Target variable (treatment duration)

## How to Run

1. Make sure you have the required libraries installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

2. Place the Excel file `Talent_Academy_Case_DT_2025.xlsx` in the same directory

3. Run the analysis script:
```python
python analysis.py
```

## Results Summary

The preprocessing pipeline successfully:
- Cleaned all missing values using appropriate strategies
- Converted text data into structured numerical features
- Created a model-ready dataset with 50+ features
- Maintained all original patient records
- Prepared data suitable for regression or classification tasks

The final dataset has no missing values and is ready for machine learning algorithms to predict treatment duration based on patient characteristics and medical conditions.

## Next Steps

The processed data could be used for:
- Predicting treatment duration for new patients
- Identifying factors that influence treatment length
- Optimizing resource allocation in rehabilitation centers
- Supporting clinical decision-making processes

## Contact

For questions about this analysis, please contact:
- Email: Sheroo16.ali@gmail.com
- LinkedIn: www.linkedin.com/in/sherwan-ali
- GitHub: www.github.com/SherwanAli0
