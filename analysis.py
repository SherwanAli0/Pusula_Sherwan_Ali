import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ==============
# Helper Functions
# ==============
def handle_missing_values(df):
    df = df.copy()
    df['Cinsiyet'].fillna(df['Cinsiyet'].mode()[0], inplace=True)
    df['Bolum'].fillna(df['Bolum'].mode()[0], inplace=True)
    df['Tanilar'].fillna('Unknown', inplace=True)
    df['UygulamaYerleri'].fillna('Unknown', inplace=True)
    df['KanGrubu'].fillna('Unknown', inplace=True)
    df['Alerji'].fillna('No known allergies', inplace=True)
    df['KronikHastalik'].fillna('None reported', inplace=True)
    return df

def feature_engineering(df):
    df = df.copy()
    df['TedaviSuresi_Clean'] = df['TedaviSuresi'].str.extract(r'(\d+)').astype(int)
    df['Age_Category'] = pd.cut(df['Yas'], bins=[0,25,45,65,100],
                                labels=['Young','Adult','Middle_Age','Senior'])
    df['Has_Chronic_Disease'] = (df['KronikHastalik'] != 'None reported').astype(int)
    df['Has_Allergy'] = (df['Alerji'] != 'No known allergies').astype(int)
    return df

def create_binary_features(df, column, prefix, top_n=5):
    df = df.copy()
    all_values = df[column].dropna().apply(lambda x: [v.strip().lower() for v in str(x).split(',')])
    all_values = pd.Series([v for sub in all_values for v in sub if v not in ['none reported','no known allergies','unknown']])
    for val in all_values.value_counts().head(top_n).index:
        col_name = f"{prefix}_{val.replace(' ','_')}"
        df[col_name] = df[column].str.lower().apply(lambda x: 1 if pd.notna(x) and val in x else 0)
    return df

def encode_and_scale(df):
    df = df.copy()
    # One-hot encode
    df = pd.get_dummies(df, columns=['Cinsiyet','Age_Category','Uyruk','KanGrubu'],
                        prefix=['Gender','Age','Country','BloodType'], drop_first=True)
    # Impute + scale age
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    df['Yas'] = imputer.fit_transform(df[['Yas']])
    df['Yas_scaled'] = scaler.fit_transform(df[['Yas']])
    return df

def preprocess_data(df):
    df = handle_missing_values(df)
    df = feature_engineering(df)
    df = create_binary_features(df, 'KronikHastalik', 'Chronic', top_n=8)
    df = create_binary_features(df, 'Alerji', 'Allergy', top_n=5)
    df = encode_and_scale(df)
    return df

# ========================
# Load and Explore Dataset
# ========================
df = pd.read_excel("Talent_Academy_Case_DT_2025.xlsx")
print(f"Dataset loaded. Shape: {df.shape}")

# Extract numeric treatment duration
df['TedaviSuresi_Clean'] = df['TedaviSuresi'].str.extract(r'(\d+)').astype(int)

# Quick summary
print(df.info())
print(df.describe())
print("Missing values:\n", df.isnull().sum())

# ========================
# Exploratory Data Analysis
# ========================
plt.figure(figsize=(12,4))
plt.subplot(1,2,1); df['TedaviSuresi_Clean'].hist(bins=20); plt.title("Treatment Duration")
plt.subplot(1,2,2); sns.boxplot(y=df['TedaviSuresi_Clean']); plt.title("Duration Boxplot")
plt.tight_layout(); plt.show()

print("Correlation (Age vs Treatment Duration):", df['Yas'].corr(df['TedaviSuresi_Clean']))

# ==================
# Run Pipeline
# ==================
df_processed = preprocess_data(df)

# Separate features and target
X = df_processed.drop(columns=['TedaviSuresi','TedaviSuresi_Clean'])
y = df_processed['TedaviSuresi_Clean']

print(f"\nFinal dataset: {X.shape[0]} samples, {X.shape[1]} features")
print("Target range:", y.min(), "-", y.max())

# Save outputs
df_processed.to_csv("processed_medical_data_complete.csv", index=False)
X.to_csv("model_features.csv", index=False)
y.to_csv("model_target.csv", index=False)
print("✅ Data preprocessing completed and files saved!")
