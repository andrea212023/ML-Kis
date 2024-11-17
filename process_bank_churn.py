import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import List, Optional, Tuple, Dict, Any

def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Drop specified columns from the dataframe.

    Args:
        df (pd.DataFrame): The raw dataframe.
        columns (List[str]): List of columns to drop.

    Returns:
        pd.DataFrame: DataFrame with specified columns dropped.
    """
    return df.drop(columns=columns)

def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataframe into training and validation sets.

    Args:
        df (pd.DataFrame): The raw dataframe.
        target_col (str): The target column.
        test_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple containing:
            - X_train (pd.DataFrame): Training data features.
            - X_val (pd.DataFrame): Validation data features.
            - y_train (pd.Series): Training data targets.
            - y_val (pd.Series): Validation data targets.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    return X_train, X_val, y_train, y_val

def identify_columns(X_train: pd.DataFrame, excluded_cols: List[str]) -> Tuple[List[str], List[str]]:
    """
    Identify numerical and categorical columns, excluding specified columns.

    Args:
        X_train (pd.DataFrame): Training data features.
        excluded_cols (List[str]): List of columns to exclude.

    Returns:
        Tuple containing:
            - List of numerical columns.
            - List of categorical columns.
    """
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    numeric_cols = [col for col in numeric_cols if col not in excluded_cols]
    categorical_cols = [col for col in categorical_cols if col not in excluded_cols]

    return numeric_cols, categorical_cols

def create_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """
    Create a preprocessor for numerical and categorical columns.

    Args:
        numeric_cols (List[str]): List of numerical columns.
        categorical_cols (List[str]): List of categorical columns.

    Returns:
        ColumnTransformer: Preprocessor for numerical and categorical columns.
    """
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    return preprocessor

def apply_preprocessor(preprocessor: ColumnTransformer, X_train: pd.DataFrame, X_val: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Optional[StandardScaler], OneHotEncoder]:
    """
    Apply the preprocessor to the training and validation data.

    Args:
        preprocessor (ColumnTransformer): The preprocessor.
        X_train (pd.DataFrame): Training data features.
        X_val (pd.DataFrame): Validation data features.

    Returns:
        Tuple containing:
            - Preprocessed training data features.
            - Preprocessed validation data features.
            - List of input columns.
            - Scaler for numeric features (if applied).
            - Encoder for categorical features.
    """
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    encoder = preprocessor.named_transformers_['cat']['onehot']
    encoded_cols = encoder.get_feature_names_out()

    input_cols = list(preprocessor.transformers[0][2]) + list(encoded_cols)
    scaler = preprocessor.named_transformers_['num']['scaler']

    return pd.DataFrame(X_train_processed, columns=input_cols), pd.DataFrame(X_val_processed, columns=input_cols), input_cols, scaler, encoder

def preprocess_data(raw_df: pd.DataFrame, excluded_cols: List[str]) -> Dict[str, Any]:
    """
    Preprocess the raw dataframe.

    Args:
        raw_df (pd.DataFrame): The raw dataframe.
        excluded_cols (List[str]): List of columns to exclude.

    Returns:
        Dict[str, Any]: Dictionary containing processed data and preprocessing objects.
    """
    df = drop_columns(raw_df, excluded_cols)
    X_train, X_val, y_train, y_val = split_data(df, 'Exited')
    numeric_cols, categorical_cols = identify_columns(X_train, excluded_cols)
    preprocessor = create_preprocessor(numeric_cols, categorical_cols)
    X_train_processed, X_val_processed, input_cols, scaler, encoder = apply_preprocessor(preprocessor, X_train, X_val)
    
    return {
        'train_X': X_train_processed,
        'train_y': y_train,
        'val_X': X_val_processed,
        'val_y': y_val,
        'input_cols': input_cols,
        'scaler': scaler,
        'encoder': encoder
    }

def preprocess_new_data(new_data: pd.DataFrame, input_cols: List[str], scaler: Optional[StandardScaler], encoder: OneHotEncoder, excluded_cols: List[str]) -> pd.DataFrame:
    """
    Preprocess new data using the fitted scaler and encoder.

    Args:
        new_data (pd.DataFrame): New data to preprocess.
        input_cols (List[str]): List of input columns.
        scaler (Optional[StandardScaler]): Fitted scaler for numeric features.
        encoder (OneHotEncoder): Fitted encoder for categorical features.
        excluded_cols (List[str]): List of columns to exclude.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    new_data = drop_columns(new_data, excluded_cols)
    numeric_cols, categorical_cols = identify_columns(new_data, excluded_cols)

    # Scale numeric features
    new_data[numeric_cols] = scaler.transform(new_data[numeric_cols])

    # Encode categorical features
    encoded_cats = encoder.transform(new_data[categorical_cols])
    encoded_cats_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))

    # Combine numeric and encoded categorical features
    new_data_processed = pd.concat([new_data[numeric_cols].reset_index(drop=True), encoded_cats_df.reset_index(drop=True)], axis=1)

    return pd.DataFrame(new_data_processed, columns=input_cols)

def main_process_data(raw_df: pd.DataFrame, excluded_cols: List[str]) -> Dict[str, Any]:
    """
    Main function to process the raw dataframe and return preprocessed data.

    Args:
        raw_df (pd.DataFrame): The raw dataframe.
        excluded_cols (List[str]): List of columns to exclude.

    Returns:
        Dict[str, Any]: Dictionary containing processed data and preprocessing objects.
    """
    return preprocess_data(raw_df, excluded_cols)
