import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import folium
from folium.plugins import HeatMap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_excel("/content/Dataset - 1.xlsx")
print("Dataset Overview:")
print(df.info())
print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\nMissing values in each column:")
print(df.isnull().sum())

df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)
df = df.convert_dtypes()
print("\nUpdated Data Types:")
print(df.dtypes)

if "Aggregate rating" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Aggregate rating"], bins=20, kde=True)
    plt.title("Distribution of Aggregate Rating")
    plt.xlabel("Aggregate Rating")
    plt.xlabel("Aggregate Rating")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.countplot(x=df["Aggregate rating"], hue=df["Aggregate rating"], palette="viridis", legend=False)
    plt.title("Class Distribution of Aggregate Rating")
    plt.xlabel("Aggregate Rating")
    plt.ylabel("Count")
    plt.show()
else:
    print("\nTarget variable 'Aggregate rating' not found in the dataset.")
print("\nBasic Statistical Measures:")
print(df.describe())

def explore_categorical(column_name):
    if column_name in df.columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(y=df[column_name], hue=df[column_name], order=df[column_name].value_counts().index[:10], palette="coolwarm", dodge=False, legend=False)
        plt.title(f"Top 10 {column_name} by Count")
        plt.xlabel("Count")
        plt.ylabel(column_name)
        plt.show()
    else:
        print(f"\nColumn '{column_name}' not found in the dataset.")
explore_categorical("Country Code")
explore_categorical("City")
explore_categorical("Cuisines")

if "Cuisines" in df.columns:
    print("\nTop 10 Cuisines with Most Restaurants:")
    print(df["Cuisines"].value_counts().head(10))
if "City" in df.columns:
    print("\nTop 10 Cities with Most Restaurants:")
    print(df["City"].value_counts().head(10))

print("\nDataset Overview:")
print(df.info())
if "Latitude" in df.columns and "Longitude" in df.columns:
    center_lat, center_lon = df["Latitude"].mean(), df["Longitude"].mean()
    restaurant_map = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=3,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.6,
        ).add_to(restaurant_map)

    restaurant_map.save("restaurants_map.html")
    print("\nRestaurant locations map saved as 'restaurants_map.html'. Open it in a browser to view.")
    heatmap = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    HeatMap(df[["Latitude", "Longitude"]].dropna()).add_to(heatmap)
    heatmap.save("restaurants_heatmap.html")
    print("Restaurant heatmap saved as 'restaurants_heatmap.html'.")
else:
    print("\nLatitude and Longitude columns not found in the dataset.")

if "City" in df.columns:
    plt.figure(figsize=(12, 6))
    top_cities = df["City"].value_counts().head(10)  # Top 10 cities with most restaurants
    sns.barplot(x=top_cities.index, y=top_cities.values, hue=top_cities.index, palette="coolwarm", legend=False)
    plt.xticks(rotation=45)
    plt.title("Top 10 Cities with Most Restaurants")
    plt.xlabel("City")
    plt.ylabel("Number of Restaurants")
    plt.show()
else:
    print("\nCity column not found in the dataset.")

if "Country Code" in df.columns:
    plt.figure(figsize=(12, 6))
    top_countries = df["Country Code"].value_counts().head(10)  # Top 10 countries
    sns.barplot(x=top_countries.index, y=top_countries.values, hue=top_countries.index, palette="viridis", legend=False)
    plt.xticks(rotation=45)
    plt.title("Top 10 Countries with Most Restaurants")
    plt.xlabel("Country Code")
    plt.ylabel("Number of Restaurants")
    plt.show()
else:
    print("\nCountry Code column not found in the dataset.")

if "Latitude" in df.columns and "Longitude" in df.columns and "Aggregate rating" in df.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df["Longitude"], y=df["Latitude"], hue=df["Aggregate rating"], palette="coolwarm", alpha=0.7)
    plt.title("Restaurant Location vs. Rating")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(title="Rating")
    plt.show()
    correlation = df[["Latitude", "Longitude", "Aggregate rating"]].corr()
    print("\nCorrelation between Location and Rating:")
    print(correlation)
else:
    print("\nLatitude, Longitude, or Aggregate rating column missing from the dataset.")

required_columns = ["Has Table booking", "Has Online delivery", "Aggregate rating", "Price range"]
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    print(f"\nMissing columns in dataset: {missing_cols}")
else:
    df["Has Table booking"] = df["Has Table booking"].astype(str).str.lower()
    df["Has Online delivery"] = df["Has Online delivery"].astype(str).str.lower()

    table_booking_counts = df["Has Table booking"].value_counts(normalize=True) * 100
    online_delivery_counts = df["Has Online delivery"].value_counts(normalize=True) * 100
    print("\nPercentage of Restaurants Offering Table Booking:")
    print(table_booking_counts)
    print("\nPercentage of Restaurants Offering Online Delivery:")
    print(online_delivery_counts)

    plt.figure(figsize=(6, 5))
    sns.barplot(
        x=table_booking_counts.index,
        y=table_booking_counts.values,
        hue=table_booking_counts.index,
        palette="coolwarm",
        legend=False
    )
    plt.title("Percentage of Restaurants Offering Table Booking")
    plt.xlabel("Table Booking Available")
    plt.ylabel("Percentage")
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.barplot(
        x=online_delivery_counts.index,
        y=online_delivery_counts.values,
        hue=online_delivery_counts.index,
        palette="viridis",
        legend=False
    )
    plt.title("Percentage of Restaurants Offering Online Delivery")
    plt.xlabel("Online Delivery Available")
    plt.ylabel("Percentage")
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.boxplot(
        x="Has Table booking",
        y="Aggregate rating",
        data=df,
        hue="Has Table booking",
        palette="coolwarm",
        legend=False
    )
    plt.title("Comparison of Ratings: Table Booking vs. No Table Booking")
    plt.xlabel("Table Booking Available")
    plt.ylabel("Average Rating")
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.countplot(x="Price range", hue="Has Online delivery", data=df, palette="magma")
    plt.title("Online Delivery Availability by Price Range")
    plt.xlabel("Price Range")
    plt.ylabel("Number of Restaurants")
    plt.legend(title="Online Delivery")
    plt.show()

required_columns = ["Price range", "Aggregate rating"]
missing_cols = [col for col in required_columns if col not in df.columns]

if missing_cols:
    print(f"\nMissing columns in dataset: {missing_cols}")
else:
    most_common_price_range = df["Price range"].mode()[0]
    print(f"\nMost Common Price Range: {most_common_price_range}")
    avg_ratings_per_price = df.groupby("Price range")["Aggregate rating"].mean().sort_values(ascending=False)
    print("\nAverage Ratings per Price Range:")
    print(avg_ratings_per_price)
    highest_rated_price = avg_ratings_per_price.idxmax()
    highest_rating_value = avg_ratings_per_price.max()

    colors = sns.color_palette("coolwarm", len(avg_ratings_per_price))
    color_mapping = {price: color for price, color in zip(avg_ratings_per_price.index, colors)}
    highest_rated_color = color_mapping[highest_rated_price]
    print(f"\nHighest Rated Price Range: {highest_rated_price} with Average Rating: {highest_rating_value:.2f}")
    print(f"Color representing the highest rating: {highest_rated_color}")

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=avg_ratings_per_price.index,
        y=avg_ratings_per_price.values,
        hue=avg_ratings_per_price.index,
        palette=colors,
        legend=False
    )
    plt.title("Average Ratings Across Different Price Ranges")
    plt.xlabel("Price Range")
    plt.ylabel("Average Rating")
    plt.xticks(rotation=0)
    plt.show()

print("\nBefore Feature Engineering:")
print(df.head())
required_columns = ["Restaurant Name", "Address", "Has Table booking", "Has Online delivery"]
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    print(f"\nMissing columns in dataset: {missing_cols}")
else:
    df["Restaurant Name Length"] = df["Restaurant Name"].apply(lambda x: len(str(x)))
    df["Address Length"] = df["Address"].apply(lambda x: len(str(x)))
    df["Has Table booking"] = df["Has Table booking"].astype(str).str.lower().map({"yes": 1, "no": 0})
    df["Has Online delivery"] = df["Has Online delivery"].astype(str).str.lower().map({"yes": 1, "no": 0})
    df.fillna({"Has Table booking": 0, "Has Online delivery": 0}, inplace=True)

    print("\nAfter Feature Engineering:")
    print(df.head())
    print("\nUpdated Data Types:")
    print(df.dtypes)

print("Dataset Overview:")
print(df.info())
df = df.dropna(subset=["Aggregate rating"])
df["Restaurant Name Length"] = df["Restaurant Name"].apply(lambda x: len(str(x)))
df["Address Length"] = df["Address"].apply(lambda x: len(str(x)))

df["Has Table booking"] = df["Has Table booking"].astype(str).str.lower().map({"yes": 1, "no": 0})
df["Has Online delivery"] = df["Has Online delivery"].astype(str).str.lower().map({"yes": 1, "no": 0})
df.fillna({"Has Table booking": 0, "Has Online delivery": 0}, inplace=True)
features = ["Price range", "Has Table booking", "Has Online delivery", "Restaurant Name Length", "Address Length"]
target = "Aggregate rating"

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {"RMSE": rmse, "R² Score": r2}

results_df = pd.DataFrame(results).T
print("\nModel Performance Comparison:")
print(results_df)

plt.figure(figsize=(8, 5))
sns.barplot(x=results_df.index, y=results_df["R² Score"], hue=results_df.index, legend=False, palette="coolwarm")
plt.title("Model R² Score Comparison")
plt.ylabel("R² Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.show()

df.columns = df.columns.str.strip()
print("Columns in dataset:", df.columns)
df.rename(columns={'Cuisines': 'Cuisine', 'Aggregate rating': 'Rating'}, inplace=True)
required_columns = ['Cuisine', 'Rating', 'Votes']

df = df[[col for col in required_columns if col in df.columns]]
display(df.head())
df.dropna(inplace=True)
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
df_expanded = df.assign(Cuisine=df['Cuisine'].str.split(', ')).explode('Cuisine')
cuisine_ratings = df_expanded.groupby('Cuisine')['Rating'].mean().sort_values(ascending=False)
cuisine_votes = df_expanded.groupby('Cuisine')['Votes'].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(y=cuisine_ratings.index[:10], x=cuisine_ratings.values[:10], hue=cuisine_ratings.index[:10], palette='viridis', legend=False)
plt.xlabel('Average Rating')
plt.ylabel('Cuisine Type')
plt.title('Top 10 Cuisines by Average Rating')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(y=cuisine_votes.index[:10], x=cuisine_votes.values[:10], hue=cuisine_votes.index[:10], palette='magma', legend=False)
plt.xlabel('Total Votes')
plt.ylabel('Cuisine Type')
plt.title('Top 10 Most Popular Cuisines Based on Votes')
plt.show()

high_rating_cuisines = cuisine_ratings[cuisine_ratings > cuisine_ratings.mean()]
print("Cuisines that tend to receive higher ratings:")
display(high_rating_cuisines)

df.columns = df.columns.str.strip()
print("Columns in dataset:", df.columns)
column_mapping = {'Aggregate rating': 'Rating', 'Ratings': 'Rating'}
df.rename(columns=column_mapping, inplace=True)

if 'Rating' not in df.columns:
    raise KeyError("The dataset does not contain a 'Rating' column. Please check the column names: ", df.columns)
df.dropna(subset=['Rating'], inplace=True)
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

plt.figure(figsize=(8, 5))
sns.histplot(df['Rating'], bins=20, kde=True, color='skyblue')
plt.xlabel('Ratings')
plt.ylabel('Frequency')
plt.title('Distribution of Ratings')
plt.show()

if 'Cuisine' in df.columns:
    df_expanded = df.assign(Cuisine=df['Cuisine'].str.split(', ')).explode('Cuisine')
    cuisine_ratings = df_expanded.groupby('Cuisine')['Rating'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(y=cuisine_ratings.index[:10], x=cuisine_ratings.values[:10], hue=cuisine_ratings.index[:10], palette='coolwarm', legend=False)
    plt.xlabel('Average Rating')
    plt.ylabel('Cuisine Type')
    plt.title('Top 10 Cuisines by Average Rating')
    plt.show()

if 'City' in df.columns:
    city_ratings = df.groupby('City')['Rating'].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(y=city_ratings.index[:10], x=city_ratings.values[:10], hue=city_ratings.index[:10], palette='magma', legend=False)
    plt.xlabel('Average Rating')
    plt.ylabel('City')
    plt.title('Top 10 Cities by Average Rating')
    plt.show()

if 'Votes' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df['Votes'], y=df['Rating'], alpha=0.6, color='purple')
    plt.xlabel('Votes')
    plt.ylabel('Rating')
    plt.title('Votes vs Rating')
    plt.show()

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Features')
plt.show()