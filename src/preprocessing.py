import pandas as pd
import logging

from src.config import CLEANED_DATA_FILE

logger = logging.getLogger(__name__)


# -------------------------------------------------
# Remove duplicate rows
# -------------------------------------------------

def remove_duplicates(df):

    before = df.shape[0]
    df     = df.drop_duplicates()
    after  = df.shape[0]

    logger.info(f"Removed {before - after} duplicate rows")

    return df


# -------------------------------------------------
# Handle missing values
# -------------------------------------------------

def handle_missing_values(df):

    logger.info("Handling missing values")

    # Drop rows without labels
    df = df.dropna(subset=["label"])

    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


# -------------------------------------------------
# Fix data types
# -------------------------------------------------

def fix_data_types(df):

    logger.info("Fixing column data types")

    df["year"]   = df["year"].astype(int)
    df["day"]    = df["day"].astype(int)
    df["length"] = df["length"].astype(int)
    df["weight"] = df["weight"].astype(float)
    df["income"] = df["income"].astype(float)

    return df


# -------------------------------------------------
# Remove invalid values
# -------------------------------------------------

def remove_invalid_values(df):

    logger.info("Removing invalid rows")

    df = df[df["income"]    >= 0]
    df = df[df["neighbors"] >= 0]
    df = df[df["count"]     >= 0]

    return df


# -------------------------------------------------
# Save cleaned dataset
# -------------------------------------------------

def save_clean_data(df):

    logger.info(f"Saving cleaned dataset to {CLEANED_DATA_FILE}")

    df.to_parquet(CLEANED_DATA_FILE, index=False)


# -------------------------------------------------
# Full preprocessing pipeline
# -------------------------------------------------

def preprocess_data(df):

    logger.info("Starting preprocessing pipeline")

    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = fix_data_types(df)
    df = remove_invalid_values(df)

    save_clean_data(df)

    logger.info("Preprocessing complete")

    return df