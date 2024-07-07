import matplotlib.pyplot as plt
import pandas as pd

# Sample data representing the distribution of different LULC classes
data = {
    'LULC_Class': ['Water', 'Vegetation', 'Agriculture', 'Settlement', 'Barren'],
    'Area (sq km)': [50, 200, 150, 100, 25]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Plotting the bar graph
plt.figure(figsize=(10, 6))
plt.bar(df['LULC_Class'], df['Area (sq km)'], color=['blue', 'green', 'yellow', 'red', 'brown'])

# Adding titles and labels
plt.title('Distribution of LULC Classes')
plt.xlabel('LULC Class')
plt.ylabel('Area (sq km)')

# Show the plot
plt.show()

