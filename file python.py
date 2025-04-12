# =============================================================================
# Section 1: Import Libraries and Set Plot Style
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# Set style for all plots
sns.set(style="whitegrid")

# =============================================================================
# Section 2: Load and Inspect the Data
# =============================================================================

# Load the dataset (update the path to your CSV file)
df = pd.read_csv(r'C:\Users\ASUS\Desktop\Spotify_2024_Global_Streaming_Data.csv')

# Basic information and summary
df.info()
print(df.describe())

# =============================================================================
# Section 3: Data Cleaning and Preparation
# =============================================================================

# Check for missing values and duplicates
print("Missing Values:\n", df.isnull().sum())
df = df.dropna()
df = df.drop_duplicates()

# Clean column names by stripping spaces and removing unwanted characters
df.columns = df.columns.str.strip().str.replace(r"[()]", "", regex=True)
df.rename(columns={
    'Avg Stream Duration Min': 'Avg_Stream_Duration_Min',
    'Monthly Listeners Millions': 'Monthly_Listeners_Millions',
    'Total Streams Millions': 'Total_Streams_Millions',
    'Total Hours Streamed Millions': 'Total_Hours_Streamed_Millions',
    'Streams Last 30 Days Millions': 'Streams_Last_30_Days_Millions',
    'Skip Rate %': 'Skip_Rate_Percent'
}, inplace=True)

# Convert columns to numeric values
cols_to_convert = [
    'Monthly_Listeners_Millions',
    'Total_Streams_Millions',
    'Total_Hours_Streamed_Millions',
    'Avg_Stream_Duration_Min',
    'Streams_Last_30_Days_Millions',
    'Skip_Rate_Percent'
]
df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

# Create an additional feature: Streams per Hour
df['Streams_per_Hour'] = df['Total_Streams_Millions'] / df['Total_Hours_Streamed_Millions']

# =============================================================================
# Section 4: Visualization - Bar Charts
# =============================================================================

# 4.1 Bar Plot: Top 10 Streamed Artists (Total Streams)
top_artists = df.groupby('Artist')['Total_Streams_Millions'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 7))
sns.barplot(x=top_artists.values, y=top_artists.index, palette='viridis')
plt.title("Top 10 Streamed Artists (Total Streams in Millions)", fontsize=16)
plt.xlabel("Total Streams (Millions)", fontsize=14)
plt.ylabel("Artist", fontsize=14)
for index, value in enumerate(top_artists.values):
    plt.text(value + max(top_artists.values)*0.01, index, f'{value:.1f}', va='center', fontsize=12)
plt.tight_layout()
plt.show()

# 4.2 Bar Plot: Total Streams by Country
country_streams = df.groupby('Country')['Total_Streams_Millions'].sum().sort_values(ascending=False)
plt.figure(figsize=(14, 7))
sns.barplot(x=country_streams.values, y=country_streams.index, palette='coolwarm')
plt.title("Total Streams by Country", fontsize=16)
plt.xlabel("Total Streams (Millions)", fontsize=14)
plt.ylabel("Country", fontsize=14)
for index, value in enumerate(country_streams.values):
    plt.text(value + max(country_streams.values)*0.005, index, f'{value:.1f}', va='center', fontsize=12)
plt.tight_layout()
plt.show()

# =============================================================================
# Section 5: Visualization - Box Plots and Pie/Donut Charts
# =============================================================================

# 5.1 Box Plot: Total Streams by Platform Type
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Platform Type', y='Total_Streams_Millions', palette='Set2')
plt.title("Total Streams by Platform Type", fontsize=16)
plt.xlabel("Platform Type", fontsize=14)
plt.ylabel("Total Streams (Millions)", fontsize=14)
plt.tight_layout()
plt.show()

# 5.2 Pie Chart: Top 5 Genres by Total Streams
top_genres = df.groupby('Genre')['Total_Streams_Millions'].sum().sort_values(ascending=False).head(5)
plt.figure(figsize=(7, 7))
plt.pie(top_genres.values, labels=top_genres.index, autopct='%1.1f%%', startangle=140, 
        colors=sns.color_palette('pastel'))
plt.title("Top 5 Genres by Total Streams", fontsize=16)
plt.tight_layout()
plt.show()

# 5.3 Pie Chart: Artist Distribution by Country
country_counts = df['Country'].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(country_counts, labels=country_counts.index, autopct='%1.1f%%', startangle=90,
        colors=sns.color_palette('Set3'))
plt.title('Artist Distribution by Country', fontsize=16)
plt.axis('equal')
plt.show()

# =============================================================================
# Section 6: Visualization - Scatter Plots, Histograms, and Heatmap
# =============================================================================

# 6.1 Scatter Plot: Skip Rate vs Average Stream Duration (colored by Genre)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Avg_Stream_Duration_Min', y='Skip_Rate_Percent', hue='Genre', palette='Set2', s=100)
plt.title('Skip Rate vs Average Stream Duration', fontsize=16)
plt.xlabel('Avg Stream Duration (Min)', fontsize=14)
plt.ylabel('Skip Rate (%)', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 6.2 Histogram: Distribution of Monthly Listeners
plt.figure(figsize=(8, 5))
sns.histplot(df['Monthly_Listeners_Millions'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Monthly Listeners', fontsize=16)
plt.xlabel('Monthly Listeners (Millions)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tight_layout()
plt.show()

# 6.3 Histogram: Distribution of Total Hours Streamed
plt.figure(figsize=(8, 5))
sns.histplot(df['Total_Hours_Streamed_Millions'], bins=20, kde=True, color='salmon')
plt.title('Distribution of Total Hours Streamed', fontsize=16)
plt.xlabel('Total Hours Streamed (Millions)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tight_layout()
plt.show()

# 6.4 Heatmap: Correlation of Key Numeric Features
plt.figure(figsize=(10, 6))
sns.heatmap(df[[
    'Monthly_Listeners_Millions', 'Total_Streams_Millions',
    'Total_Hours_Streamed_Millions', 'Avg_Stream_Duration_Min',
    'Streams_Last_30_Days_Millions', 'Skip_Rate_Percent'
]].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap', fontsize=16)
plt.tight_layout()
plt.show()

# =============================================================================
# Section 7: Additional Bar Charts and Derived Metrics
# =============================================================================

# 7.1 Bar Chart: Top 10 Countries by Total Streams (Alternative Orientation)
top_countries = df.groupby('Country')['Total_Streams_Millions'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_countries.index, y=top_countries.values, palette='viridis')
plt.title('Top 10 Countries by Total Streams', fontsize=16)
plt.xlabel('Country', fontsize=14)
plt.ylabel('Total Streams (Millions)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 7.2 Bar Chart: Streams per Hour Efficiency by Artist (Top 10)
plt.figure(figsize=(10, 5))
sns.barplot(data=df.sort_values(by='Streams_per_Hour', ascending=False).head(10),
            x='Artist', y='Streams_per_Hour', palette='cubehelix')
plt.title('Top 10 Artists: Streams per Hour Efficiency', fontsize=16)
plt.ylabel('Streams per Hour', fontsize=14)
plt.xlabel('Artist', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
