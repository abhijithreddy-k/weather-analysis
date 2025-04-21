# weather-analysis
nalyzed Madrid daily weather (1997–2015) using Python. Covered EDA, data cleaning, visualizations, and statistical tests. Explored temperature trends, humidity, pressure correlations, and rain patterns. Dataset from https://mavenanalytics.io/data-playground.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt   
import scipy.stats as stats
from scipy.stats import norm

df = pd.read_csv(r"D:/SEM 4/INT 375/Python Project/Madrid Daily Weather 1997-2015.csv",encoding="Iso-8859-1")

# Complete EDA and Data Cleaning

# View the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())



#  Display the shape of the dataset
print("\nShape of the dataset (rows, columns):")
print(df.shape)

#  View column names
print("\nColumn names:")
print(df.columns)

# View index range
print("\nIndex of the DataFrame:")
print(df.index)

# Check data types
print("\nData types of each column:")
print(df.dtypes)

# Summary statistics
print("\nSummary statistics:")
print(df.describe(include='all'))

# Check missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Drop rows with missing values (temporary cleaning)
df_dropped = df.dropna()
print("\nShape after dropping missing values:")
print(df_dropped.shape)
# 
df = df.bfill()
print("\nMissing values after backward fill:")
print(df.isnull().sum())

#  Drop unwanted columns (like unnamed index columns, if present)
if 'unnamed: 0' in df.columns:
    df = df.drop(columns=['unnamed: 0'])

# Final shape after cleaningprint
print("\nFinal shape of cleaned DataFrame:")
print(df.shape)


#Check for duplicates
print("\nDuplicate rows in the dataset:")
print(df.duplicated().sum())

# Drop duplicates
df = df.drop_duplicates()
print("\nShape after dropping duplicates:")
print(df.shape)



# Objective 1 To conduct statistical analysis on weather changes
#Mean of Max TemperatureC
print(f"The mean of Max TemperatureC is {round(df['Max TemperatureC'].mean(), 2)}°C.")

#min sea level and last occured
min_pressure = df[' Min Sea Level PressurehPa'].min()
min_rows = df[df[' Min Sea Level PressurehPa'] == min_pressure]
last_date = min_rows['CET'].iloc[-1]
print(f"The minimum Sea Level PressurehPa is {min_pressure} hPa and it last occurred on {last_date}.")

#Max Humidity and its last Occurance
max_humidity = df['Max Humidity'].max()
max_rows = df[df['Max Humidity'] == max_humidity]
last_date = max_rows['CET'].iloc[-1]
print(f"The maximum Max Humidity is {max_humidity}% and it last occurred on {last_date}.")

#std of WindDirDegrees
std_wind_dir = df['WindDirDegrees'].std()
print(f"The standard deviation of Wind Direction Degrees is {round(std_wind_dir, 2)}°.")

#Adding a new column for temperature range and exporting to excel
df['Temp Range'] = df['Max TemperatureC'] - df['Min TemperatureC']
print("\nNew column 'Temp Range' added successfully!")
print(df[['CET', 'Max TemperatureC', 'Min TemperatureC', 'Temp Range']].head())
# Select only the relevant columns
temp_range_df = df[['CET', 'Max TemperatureC', 'Min TemperatureC', 'Temp Range']]
# Export to Excel file
temp_range_df.to_excel("temperature_range_analysis.xlsx", index=False)
print("Temperature range data has been saved to 'temperature_range_analysis.xlsx'")


#Groupby and mean of Max TemperatureC by Year and Month
# Step 1: Make sure 'CET' is datetime
df['CET'] = pd.to_datetime(df['CET'])
# Step 2: Create 'Year' and 'Month' columns
df['Year'] = df['CET'].dt.year
df['Month'] = df['CET'].dt.month
# Step 3: Filter data for the year 2006
df_2006 = df[df['Year'] == 2006]
# Step 4: Group by Month and calculate average Max Temperature
monthly_avg_temp_2006 = df_2006.groupby('Month')['Max TemperatureC'].mean()
print("\nAverage Max Temperature for each month in 2006:")
print(monthly_avg_temp_2006)








# Objective 2 Daily & Monthly Weather Patterns
mean_temp = df['Mean TemperatureC'].dropna()

# Count how many times each temperature occurs (frequency)
temp_counts = mean_temp.value_counts().sort_index()

# Plot line chart
plt.figure(figsize=(10, 6))
plt.plot(temp_counts.index, temp_counts.values, marker='o', linestyle='-', color='darkorange')
plt.xlabel('Mean Temperature (°C)')
plt.ylabel('Number of Days')
plt.title('Number of Days for Each Mean Temperature')
plt.grid(True)
plt.tight_layout()
plt.show()







# objective 3 Correlation Between Weather Parameters

df = pd.read_csv(r"D:\SEM 4\INT 375\Python Project\Madrid Daily Weather 1997-2015.csv",encoding="Iso-8859-1")
pressure_temp_corr = df[[
    ' Mean Sea Level PressurehPa',  # With leading space
    'Max TemperatureC',
    'Mean TemperatureC', 
    'Min TemperatureC'
]].corr()

# Create the heatmap with all proper formatting
plt.figure(figsize=(8, 6))
sns.heatmap(
    pressure_temp_corr,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    mask=np.triu(np.ones_like(pressure_temp_corr))  # Upper triangle mask
)

# Add titles and formatting
plt.title("Pressure vs. Temperature Correlation Matrix", pad=20, fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()

# Show the plot
plt.show()






# Select only the two columns
selected_data = df[['Mean TemperatureC', ' Mean Humidity']]

# Calculate the correlation matrix
corr_matrix = selected_data.corr()

# Plot the heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation between Mean Temperature and Mean Humidity')
plt.tight_layout()
plt.show()







# objective 4 temperature distribution on rainy vs non-rainy days




#Step 1: Ensure 'CET' is datetime
df['CET'] = pd.to_datetime(df['CET'])

# Step 2: Clean column names if needed
if ' Events' in df.columns:
    df.rename(columns={' Events': 'Events'}, inplace=True)

# Step 3: Create rainy and non-rainy subsets
rainy_days = df[df['Events'].str.contains('Rain', na=False)]
non_rainy_days = df[~df['Events'].str.contains('Rain', na=False)]

# Step 4: Plot histograms
plt.figure(figsize=(10, 6))
plt.hist(rainy_days['Mean TemperatureC'], bins=20, alpha=0.6, color='blue', label='Rainy Days')
plt.hist(non_rainy_days['Mean TemperatureC'], bins=20, alpha=0.6, color='orange', label='Non-Rainy Days')
plt.xlabel('Mean Temperature (°C)')
plt.ylabel('Number of Days')
plt.title('Temperature Distribution on Rainy vs Non-Rainy Days')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()













# Objective 5 To visualize trends in temperature, rainfall, and other variables


# Average Air prussure before Rainy Days
# Clean column names
df.columns = df.columns.str.strip()
# Convert CET to datetime
df['CET'] = pd.to_datetime(df['CET'])
# Sort by date
df = df.sort_values(by='CET')
# Filter Rainy Days (SAFELY)
rainy_days = df[df['Events'].str.contains('Rain', na=False)].copy()

# Now it's safe to modify without warnings
rainy_days['CET'] = pd.to_datetime(rainy_days['CET'])

# Now safely calculate pressure trend
pressure_trend = []
for rain_date in rainy_days['CET']:
    rain_date = pd.to_datetime(rain_date)  # Ensure datetime format
    prev_days = df[(df['CET'] >= rain_date - pd.Timedelta(days=3)) & (df['CET'] < rain_date)]
    pressure_trend.append(prev_days['Mean Sea Level PressurehPa'].values)
# Create DataFrame
pressure_df = pd.DataFrame(pressure_trend)
pressure_df.columns = ['3 days before', '2 days before', '1 day before']
# Avrage pressure
avg_pressure = pressure_df.mean()


# Plotting
plt.figure(figsize=(8, 5))
plt.plot(['3 days before', '2 days before', '1 day before'], avg_pressure, marker='o', color='green')
plt.title('Average Sea Level Pressure Before Rainy Days')
plt.xlabel('Days Before Rain')  
plt.ylabel('Average Pressure (hPa)')
plt.grid(True)
plt.tight_layout()
plt.show()


df.columns = df.columns.str.strip()
df['CET'] = pd.to_datetime(df['CET'])  # Just to be safe
df['Month'] = df['CET'].dt.month_name()

rainy_days = df[df['Events'].str.contains('Rain', na=False)]
plt.figure(figsize=(8, 6))
sns.scatterplot(data=rainy_days, x='Mean Sea Level PressurehPa', y='Mean TemperatureC', hue='Month', palette='coolwarm')
plt.title('Temperature vs Sea Level Pressure on Rainy Days')
plt.xlabel('Mean Sea Level Pressure (hPa)')
plt.ylabel('Mean Temperature (°C)')
plt.xticks(rotation=15)
plt.tight_layout()
plt.grid(True)
plt.show()





# Objective 6 Compare mean temperature in winter months vs summer months. 
#Groupby and mean of Max TemperatureC by Year and Month
# Make sure 'CET' is datetime
df['CET'] = pd.to_datetime(df['CET'])
# Create 'Year' and 'Month' columns
df['Year'] = df['CET'].dt.year
df['Month'] = df['CET'].dt.month

# Clean column names and convert date
df.columns = df.columns.str.strip()
df['CET'] = pd.to_datetime(df['CET'])
df['Month'] = df['CET'].dt.month

# Define winter and summer months
winter_months = [12, 1, 2]
summer_months = [6, 7, 8]

# Extract temperature data
winter_temps = df[df['Month'].isin(winter_months)]['Mean TemperatureC'].dropna()
summer_temps = df[df['Month'].isin(summer_months)]['Mean TemperatureC'].dropna()

# Calculate statistics
winter_mean = winter_temps.mean()
summer_mean = summer_temps.mean()
winter_std = winter_temps.std()
summer_std = summer_temps.std()
n_winter = len(winter_temps)
n_summer = len(summer_temps)

# Two-sample Z-test
z = (summer_mean - winter_mean) / np.sqrt((summer_std**2 / n_summer) + (winter_std**2 / n_winter))
p_value = 2 * (1 - norm.cdf(abs(z)))  # Two-tailed

# Print results
print("Winter Mean Temperature:", round(winter_mean, 2), "°C")
print("Summer Mean Temperature:", round(summer_mean, 2), "°C")
print("Z-score:", round(z, 2))
print("P-value:", p_value)








"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df.columns = df.columns.str.strip()  # <== strip ALL whitespace from column names early
df = df.bfill()  

# Objective 2: Line Chart - Frequency of Mean Temperature
mean_temp = df['Mean TemperatureC'].dropna()
temp_counts = mean_temp.value_counts().sort_index()

# Objective 3: Correlation matrices
pressure_temp_corr = df[['Mean Sea Level PressurehPa', 'Max TemperatureC', 'Mean TemperatureC', 'Min TemperatureC']].corr()
humidity_corr = df[['Mean TemperatureC', 'Mean Humidity']].corr()

# Objective 4: Rainy vs Non-Rainy Temperature Histograms
df['CET'] = pd.to_datetime(df['CET'])
if 'Events' not in df.columns:
    df.rename(columns={' Events': 'Events'}, inplace=True)

rainy_days = df[df['Events'].str.contains('Rain', na=False)]
non_rainy_days = df[~df['Events'].str.contains('Rain', na=False)]

# Objective 5: Pressure Trend Before Rainy Days
df = df.sort_values(by='CET')
pressure_trend = []
for rain_date in rainy_days['CET']:
    rain_date = pd.to_datetime(rain_date)
    prev_days = df[(df['CET'] >= rain_date - pd.Timedelta(days=3)) & (df['CET'] < rain_date)]
    pressure_trend.append(prev_days['Mean Sea Level PressurehPa'].values)

pressure_df = pd.DataFrame(pressure_trend)
pressure_df.columns = ['3 days before', '2 days before', '1 day before']
avg_pressure = pressure_df.mean()

# Start plotting in subplots
fig, axs = plt.subplots(3, 2, figsize=(16, 18))

# Plot 1: Line Chart of Mean Temp Frequencies
axs[0, 0].plot(temp_counts.index, temp_counts.values, marker='o', linestyle='-', color='darkorange')
axs[0, 0].set_title('Number of Days for Each Mean Temperature')
axs[0, 0].set_xlabel('Mean Temperature (°C)')
axs[0, 0].set_ylabel('Number of Days')
axs[0, 0].grid(True)

# Plot 2: Pressure vs Temperature Correlation Heatmap
sns.heatmap(pressure_temp_corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axs[0, 1])
axs[0, 1].set_title('Pressure vs Temperature Correlation')

# Plot 3: Temp vs Humidity Correlation
sns.heatmap(humidity_corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axs[1, 0])
axs[1, 0].set_title('Temperature vs Humidity Correlation')

# Plot 4: Rainy vs Non-Rainy Temperature Distribution
axs[1, 1].hist(rainy_days['Mean TemperatureC'], bins=20, alpha=0.6, color='blue', label='Rainy Days')
axs[1, 1].hist(non_rainy_days['Mean TemperatureC'], bins=20, alpha=0.6, color='orange', label='Non-Rainy Days')
axs[1, 1].set_title('Temperature Distribution on Rainy vs Non-Rainy Days')
axs[1, 1].set_xlabel('Mean Temperature (°C)')
axs[1, 1].set_ylabel('Number of Days')
axs[1, 1].legend()
axs[1, 1].grid(True)

# Plot 5: Average Pressure Before Rainy Days
axs[2, 0].plot(['3 days before', '2 days before', '1 day before'], avg_pressure, marker='o', color='green')
axs[2, 0].set_title('Average Pressure Before Rainy Days')
axs[2, 0].set_xlabel('Days Before Rain')
axs[2, 0].set_ylabel('Average Pressure (hPa)')
axs[2, 0].grid(True)

# Hide unused subplot (bottom right)
axs[2, 1].axis('off')

# Layout adjustment
plt.tight_layout()
plt.show()

# Final Plot (Separate): Scatterplot with Hue (Month)
df['Month'] = df['CET'].dt.month_name()
rainy_days = df[df['Events'].str.contains('Rain', na=False)]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=rainy_days, x='Mean Sea Level PressurehPa', y='Mean TemperatureC', hue='Month', palette='coolwarm')
plt.title('Temperature vs Sea Level Pressure on Rainy Days')
plt.xlabel('Mean Sea Level Pressure (hPa)')
plt.ylabel('Mean Temperature (°C)')
plt.xticks(rotation=15)
plt.grid(True)
plt.tight_layout()
plt.show()

"""


