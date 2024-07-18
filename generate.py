
import pandas as pd
import numpy as np

# Generating sample data
np.random.seed(0)
rows = 1000

data = {
    'Date': pd.date_range(start='1/1/2020', periods=rows, freq='D'),
    'Product': np.random.choice(['Product A', 'Product B', 'Product C'], size=rows),
    'Sales': np.random.uniform(100, 5000, size=rows).round(2),
    'Quantity': np.random.randint(1, 100, size=rows)
}

# Creating DataFrame
df = pd.DataFrame(data)

# Save to a CSV file
df.to_csv('sample_sales_data.csv', index=False)

df.head()