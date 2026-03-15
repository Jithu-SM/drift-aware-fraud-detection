"""
encode_labels.py  — Pre-fit and save LabelEncoders for categorical columns.
----------------
This is your original script, preserved exactly.
Run this once if you need stand-alone encoders separate from train.py.
"""

import joblib
from sklearn.preprocessing import LabelEncoder
from constants import PAYMENT_TYPES, DEVICE_OS

payment_encoder = LabelEncoder()
device_encoder  = LabelEncoder()

payment_encoder.fit(PAYMENT_TYPES)
device_encoder.fit(DEVICE_OS)

encoders = {
    "payment_type": payment_encoder,
    "device_os":    device_encoder,
}

joblib.dump(encoders, "models/encoders.pkl")
print("Encoders saved successfully")