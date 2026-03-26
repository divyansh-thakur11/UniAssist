from sklearn.preprocessing import LabelEncoder

def preprocess_student(df):
    df = df.copy()

    # Encode categorical columns
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df