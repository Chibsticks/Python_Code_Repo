# Requirements
!pip install openpyxl pandas

## How to run:
## 1. Make sure your data matches the size, shape and headers in the sample data 'VENDOR_CURATION_LOTTERY.xlsx'. You can have as many rows as needed.
## 2. You may want to amend the category_constraints to add/remove/edit constraints. This feature is still being tested as I need a larger dataset to ensure it works.
## 3. The 'vendor_input' defaults to 20 (20 'Den' tables). You can overwrite this when running as it asks for your user input.
## Additional: I use Jupyter Notebooks for my code development so I can check, test, review and debug the output. This is why there are 2 different 'cells'. You don't need to use jupyter notebooks to run this script however.

import pandas as pd
import numpy as np

# Load the vendor data into a dataframe from the source database, I'm using an excel spreadsheet with two tabs
vendor_data_df = pd.read_excel('VENDOR_CURATION_LOTTERY.xlsx', sheet_name='VENDOR_DATA')

# Load the vendor categories which contains the inverse relationship and weight calculations
den_categories = pd.read_excel('VENDOR_CURATION_LOTTERY.xlsx', sheet_name='DATA_DICTIONARY')

# Extract weights for each category and filter out NaN values and unnecessary fields, weights for the categories are calculated in the database (Normalised inverse relationship)
weights = {row['FIELD']: row['WEIGHT'] for _, row in den_categories.iterrows() if not pd.isna(row['WEIGHT']) and row['FIELD'] not in ['VENDOR_ID', 'PASSED_CURATION']}

# Define the minimum and maximum percentages for each category, can be expanded as needed, example used is a min/max of new vendors set to 20%
category_constraints = {
    'NEW_VENDOR': (0.20, 0.20), # Minimum 20%, Maximum 20%
    # Add other categories as needed
    # 'CATEGORY_NAME': (min_percentage, max_percentage),
}

# DEBUG code to print weights dictionary to verify working, commented out
# print("Weights Dictionary:", weights)

# Function to normalise the probabilities for vendors who have selected multiple different categories, so they are not given an unfair advantage
def adjust_probabilities(vendors, probabilities, constraints, selected_vendors):
    for category, (min_pct, max_pct) in constraints.items():
        category_count = sum(1 for vendor in selected_vendors if vendor[category] == 'Y')
        current_pct = category_count / len(selected_vendors) if selected_vendors else 0
        if current_pct < min_pct:
            for i, vendor in enumerate(vendors):
                if vendor[category] == 'Y':
                    probabilities[i] += (min_pct - current_pct) / len(vendors)
        elif current_pct > max_pct:
            for i, vendor in enumerate(vendors):
                if vendor[category] == 'Y':
                    probabilities[i] -= (current_pct - max_pct) / len(vendors)
    probabilities = np.clip(probabilities, 0, 1)
    probabilities /= probabilities.sum()
    return probabilities

# Create a function which uses the weights to assign a user-defined number of slots to vendors
def lottery_picker(vendor_data, weights, constraints, max_vendors=20):
    vendor_scores = []

    for index, vendor in vendor_data.iterrows():
        if vendor['PASSED_CURATION'] == 'Y':
            # Calculate the total weighted score for each vendor
            total_weight = sum(weights.get(category, 0) for category in weights if vendor[category] == 'Y')
            if total_weight > 0:
                # Normalize weight for vendors selecting many categories
                total_weight /= vendor['TOTAL_CATEGORIES']
                vendor_scores.append((vendor, total_weight))

    # Error handling - Check if there are no vendors with positive weights
    if not vendor_scores:
        raise ValueError("No vendors with positive weights found")

    # Normalize the scores to get a probability distribution
    total_score = sum(score for _, score in vendor_scores)
    normalized_scores = [(vendor, score / total_score) for vendor, score in vendor_scores]

    # Extract vendor IDs and their corresponding probabilities
    vendors, probabilities = zip(*normalized_scores)

    # Select vendors based on the probabilities
    selected_vendors = []
    while len(selected_vendors) < max_vendors:
        probabilities = adjust_probabilities(vendors, list(probabilities), constraints, selected_vendors)
        selected_vendor_index = np.random.choice(len(vendors), p=probabilities)
        selected_vendor = vendors[selected_vendor_index]
        selected_vendors.append(selected_vendor)
        vendors = [v for i, v in enumerate(vendors) if i != selected_vendor_index]
        probabilities = [p for i, p in enumerate(probabilities) if i != selected_vendor_index]

    # Extract the vendor IDs from the selected vendors
    selected_vendor_ids = [vendor['VENDOR_ID'] for vendor in selected_vendors]

    return selected_vendor_ids

# User input the max number of vendors, uses 20 if null
vendor_input=input("Enter the max number of vendors (Press Enter to use default 20): ")
vendor_input_val = int(vendor_input) if vendor_input else 20
print(f"Max Vendors set to: {vendor_input_val}")

# Run the lottery picker function
selected_vendors = lottery_picker(vendor_data_df, weights, category_constraints, max_vendors=vendor_input_val)
selected_vendors_df = pd.DataFrame(selected_vendors, columns=['VENDOR_ID'])

# Save the selected vendors output into the source folder as a csv file for analysis and viewing
selected_vendors_df.to_csv('selected_vendors_output.csv', index=False)

# Displays the selected vendors in the console
selected_vendors_df