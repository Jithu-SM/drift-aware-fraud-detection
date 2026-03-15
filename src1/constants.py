PAYMENT_TYPES = [
    "CARD",
    "UPI",
    "TRANSFER",
    "CASH_OUT",
    "CASH_IN"
]

DEVICE_OS = [
    "Windows",
    "Android",
    "iOS",
    "Linux",
    "MacOS"
]


# the project is to make a bank transaction fraud detection system using xgboost that adapt and retrain the model when drift detects in the incoming data... for the retraining the model saves recent transactions and used it when the drift is detected... also admin is allowed to change the predicted resultant label fraud/legit to vice versa and it is used in retraining... help me build this project without changing the current structure and guide me through it... make sure the drift will be detected when the pattern in the incoming data changes... also generate some sample datasets and also a dataset can trigger the concept drift and retraining