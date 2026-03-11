import pandas as pd
import random
from datetime import datetime, timedelta
import numpy as np

# Read existing data
df = pd.read_csv('data/transactions.csv')

# Get the last transaction ID number
last_id = int(df['transaction_id'].iloc[-1].replace('TXN', ''))
last_step = int(df['step'].iloc[-1])
last_date = pd.to_datetime(df['timestamp'].iloc[-1])

# Type to encoded mapping
type_encoding = {
    'CASH_IN': 0.0,
    'CASH_OUT': 1.0,
    'DEBIT': 2.0,
    'PAYMENT': 3.0,
    'TRANSFER': 4.0
}

# Generate new transactions
new_transactions = []

# Scenario 1: Normal morning business transactions (9-11 AM)
for i in range(8):
    txn_id = f'TXN{last_id + 1 + i:03d}'
    step = last_step + (i + 1) * 10
    
    if i % 3 == 0:  # Payment
        txn_type = 'PAYMENT'
        amount = round(random.uniform(500, 5000), 2)
        old_balance = round(random.uniform(5000, 30000), 2)
        new_balance = old_balance - amount
        fraud_prob = random.uniform(0.00001, 0.0001)
        hour = random.choice([9, 10, 11])
    elif i % 3 == 1:  # DEBIT
        txn_type = 'DEBIT'
        amount = round(random.uniform(100, 2000), 2)
        old_balance = round(random.uniform(3000, 15000), 2)
        new_balance = old_balance - amount
        fraud_prob = random.uniform(0.00001, 0.00005)
        hour = random.choice([9, 10, 11])
    else:  # Small legitimate transfer
        txn_type = 'TRANSFER'
        amount = round(random.uniform(1000, 10000), 2)
        old_balance = round(random.uniform(10000, 50000), 2)
        new_balance = old_balance - amount
        fraud_prob = random.uniform(0.00001, 0.0001)
        hour = random.choice([9, 10, 11])
    
    new_transactions.append({
        'transaction_id': txn_id,
        'step': step,
        'type': txn_type,
        'amount': amount,
        'oldbalanceOrg': old_balance,
        'newbalanceOrig': new_balance,
        'oldbalanceDest': round(random.uniform(0, 5000), 2),
        'newbalanceDest': round(random.uniform(0, 10000), 2),
        'transaction_hour': float(hour),
        'type_encoded': type_encoding[txn_type],
        'fraud_probability': fraud_prob,
        'fraud_prediction': 'Legitimate 🟢',
        'risk_color': 'green',
        'admin_label': '',
        'timestamp': (last_date + timedelta(hours=random.randint(1, 48))).strftime('%Y-%m-%d %H:%M:%S.%f').rstrip('0')
    })

# Scenario 2: Afternoon cash withdrawals and legitimate transactions (14-17)
for i in range(6):
    txn_id = f'TXN{last_id + 9 + i:03d}'
    step = last_step + (i + 9) * 10
    
    if i % 2 == 0:  # CASH_OUT legitimate
        txn_type = 'CASH_OUT'
        amount = round(random.uniform(5000, 50000), 2)
        old_balance = round(random.uniform(50000, 150000), 2)
        new_balance = old_balance - amount
        fraud_prob = random.uniform(0.00005, 0.001)
        hour = random.choice([14, 15, 16, 17])
    else:  # CASH_IN legitimate
        txn_type = 'CASH_IN'
        amount = round(random.uniform(2000, 20000), 2)
        old_balance = round(random.uniform(1000, 10000), 2)
        new_balance = old_balance + amount
        fraud_prob = random.uniform(0.00001, 0.0001)
        hour = random.choice([14, 15, 16, 17])
    
    new_transactions.append({
        'transaction_id': txn_id,
        'step': step,
        'type': txn_type,
        'amount': amount,
        'oldbalanceOrg': old_balance,
        'newbalanceOrig': new_balance,
        'oldbalanceDest': round(random.uniform(0, 10000), 2) if txn_type == 'CASH_OUT' else 0.0,
        'newbalanceDest': round(random.uniform(10000, 50000), 2) if txn_type == 'CASH_OUT' else 0.0,
        'transaction_hour': float(hour),
        'type_encoded': type_encoding[txn_type],
        'fraud_probability': fraud_prob,
        'fraud_prediction': 'Legitimate 🟢',
        'risk_color': 'green',
        'admin_label': '',
        'timestamp': (last_date + timedelta(hours=random.randint(50, 72))).strftime('%Y-%m-%d %H:%M:%S.%f').rstrip('0')
    })

# Scenario 3: Suspicious fraudulent transactions (late night hours)
for i in range(4):
    txn_id = f'TXN{last_id + 15 + i:03d}'
    step = last_step + (i + 15) * 10
    
    txn_type = random.choice(['TRANSFER', 'CASH_OUT'])
    amount = round(random.uniform(200000, 2000000), 2)
    old_balance = amount  # Suspicious: old balance equals amount (emptying account)
    new_balance = 0.0
    fraud_prob = round(random.uniform(0.95, 0.9999), 10)
    hour = random.choice([0, 1, 2, 3, 22, 23])  # Late night
    
    new_transactions.append({
        'transaction_id': txn_id,
        'step': step,
        'type': txn_type,
        'amount': amount,
        'oldbalanceOrg': old_balance,
        'newbalanceOrig': new_balance,
        'oldbalanceDest': round(random.uniform(0, 10000), 2),
        'newbalanceDest': amount + round(random.uniform(0, 10000), 2),
        'transaction_hour': float(hour),
        'type_encoded': type_encoding[txn_type],
        'fraud_probability': fraud_prob,
        'fraud_prediction': 'Fraudulent 🔴',
        'risk_color': 'red',
        'admin_label': '',
        'timestamp': (last_date + timedelta(hours=random.randint(74, 96))).strftime('%Y-%m-%d %H:%M:%S.%f').rstrip('0')
    })

# Scenario 4: Mixed everyday transactions (various times)
for i in range(7):
    txn_id = f'TXN{last_id + 19 + i:03d}'
    step = last_step + (i + 19) * 10
    hour = float(random.randint(0, 23))
    
    txn_choices = ['PAYMENT', 'DEBIT', 'TRANSFER', 'CASH_OUT', 'CASH_IN']
    weights = [0.3, 0.25, 0.2, 0.15, 0.1]
    txn_type = random.choices(txn_choices, weights=weights)[0]
    
    if txn_type == 'PAYMENT':
        amount = round(random.uniform(100, 3000), 2)
    elif txn_type == 'DEBIT':
        amount = round(random.uniform(50, 2000), 2)
    elif txn_type == 'TRANSFER':
        amount = round(random.uniform(500, 100000), 2)
    elif txn_type == 'CASH_OUT':
        amount = round(random.uniform(1000, 80000), 2)
    else:  # CASH_IN
        amount = round(random.uniform(500, 30000), 2)
    
    old_balance = round(random.uniform(amount, amount * 3), 2)
    
    if txn_type in ['CASH_IN']:
        new_balance = old_balance + amount
    else:
        new_balance = old_balance - amount
    
    is_fraud = random.random() < 0.1  # 10% fraud rate
    
    if is_fraud:
        fraud_prob = round(random.uniform(0.92, 0.9999), 10)
        prediction = 'Fraudulent 🔴'
        color = 'red'
    else:
        fraud_prob = random.uniform(0.00001, 0.001)
        prediction = 'Legitimate 🟢'
        color = 'green'
    
    new_transactions.append({
        'transaction_id': txn_id,
        'step': step,
        'type': txn_type,
        'amount': amount,
        'oldbalanceOrg': old_balance,
        'newbalanceOrig': new_balance,
        'oldbalanceDest': round(random.uniform(0, 20000), 2) if txn_type in ['TRANSFER', 'CASH_OUT'] else 0.0,
        'newbalanceDest': round(random.uniform(0, 100000), 2) if txn_type in ['TRANSFER', 'CASH_OUT'] else 0.0,
        'transaction_hour': hour,
        'type_encoded': type_encoding[txn_type],
        'fraud_probability': fraud_prob,
        'fraud_prediction': prediction,
        'risk_color': color,
        'admin_label': '',
        'timestamp': (last_date + timedelta(hours=random.randint(100, 240))).strftime('%Y-%m-%d %H:%M:%S.%f').rstrip('0')
    })

# Create DataFrame from new transactions
new_df = pd.DataFrame(new_transactions)

# Append to existing data
result_df = pd.concat([df, new_df], ignore_index=True)

# Save to file
result_df.to_csv('data/transactions.csv', index=False)

print(f"✅ Generated {len(new_transactions)} new transactions")
print(f"Total transactions: {len(result_df)}")
print("\nGenerated scenarios:")
print("  • 8 morning business transactions (9-11 AM)")
print("  • 6 afternoon withdrawal transactions (14-17)")
print("  • 4 suspicious fraudulent late-night transactions")
print("  • 7 mixed everyday transactions with 10% fraud ratio")
print("\nNew transactions added to: data/transactions.csv")
