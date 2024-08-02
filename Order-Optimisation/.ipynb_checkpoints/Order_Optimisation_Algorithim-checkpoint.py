# Requirements
!pip install pandas random math tqdm

## How to run:
## 1. Make sure your data matches the size, shape and headers in the sample data 'VENDORS_CSV.csv'. You can have as many rows as needed.
## 2. Amend the population size and number of generations accordingly. If the script is taking too long to run, you may need to batch your data into smaller chunks. 
##    With the number of generations, more does not necessarily always = better result. The result may be marginally better (> 0.01) and for this purpose we don't need it to be
##    Scientifically accurate, given we will be reviewing the output anyway.
## 3. The 'optimal_score' defaults to 80 (80% success). You can overwrite this when running as it asks for your user input.
## 4. The script will stop running once it has met the optimal_score value threshold. If you want it to run for the entire # of generations you will need to set a higher optimal score
## Additional: I use Jupyter Notebooks for my code development so I can check, test, review and debug the output. This is why there are 3 different 'cells'. You don't need to use jupyter notebooks to run this script however.

import pandas as pd
import random
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load vendors data from CSV file
vendors_df = pd.read_csv("VENDORS_CSV.csv")

# Convert vendors_df to dictionary
vendors = vendors_df.set_index('VEND_NUMB').apply(lambda x: {
    'neighbors': [x['N_ONE'], x['N_TWO'], x['N_THREE']],
    'N1_IS_CARER': x['ONE_IS_CARER'],
    'N1_SHARE_CARD_READER': x['ONE_SHARE_CARD_READER'],
    'N2_IS_CARER': x['TWO_IS_CARER'],
    'N2_SHARE_CARD_READER': x['TWO_SHARE_CARD_READER'],
    'N3_IS_CARER': x['THREE_IS_CARER'],
    'N3_SHARE_CARD_READER': x['THREE_SHARE_CARD_READER'],
    'distance_one': x['DISTANCE_ONE'],
    'distance_two': x['DISTANCE_TWO'],
    'distance_three': x['DISTANCE_THREE']
}, axis=1).to_dict()

# Function to calculate the success rate of a given arrangement
def calculate_success_rate(arrangement, vendors):
    total_success = 0
    total_penalties = 0
    detailed_scores = []
    num_vendors_with_requests = 0

    # Check all distance constraints first to build a conflict list
    conflict_dict = {}
    for vendor in arrangement:
        conflict_dict[vendor] = []
        for other_vendor in vendors:
            if vendor in [vendors[other_vendor]['distance_one'], vendors[other_vendor]['distance_two'], vendors[other_vendor]['distance_three']]:
                conflict_dict[vendor].append(other_vendor)

    for i, vendor in enumerate(arrangement):
        confirmed_neighbors = [0, 0, 0]
        vendor_neighbors = [vn for vn in vendors[vendor]['neighbors'] if not pd.isna(vn)]
        distance_vendors = [dv for dv in [vendors[vendor]['distance_one'], vendors[vendor]['distance_two'], vendors[vendor]['distance_three']] if not pd.isna(dv)]

        # If vendor has no neighbor and no distance requests, skip
        if len(vendor_neighbors) == 0 and len(distance_vendors) == 0:
            detailed_scores.append({
                "vendor": vendor,
                "confirmed_neighbors": confirmed_neighbors,
                "neighbor_success_rate": None,
                "distance_penalty": None
            })
            continue

        num_vendors_with_requests += 1
        
        # If vendor is in conflict, ignore their neighbor requests
        if vendor in conflict_dict and conflict_dict[vendor]:
            vendor_neighbors = []  # Ignore neighbor requests if in conflict

        # Calculate neighbor success rate
        if len(vendor_neighbors) == 0:
            success_rate = 100
        else:
            if i == 0:
                if len(arrangement) > 1 and arrangement[i + 1] in vendor_neighbors:
                    index = vendor_neighbors.index(arrangement[i + 1])
                    confirmed_neighbors[index] = 1
                elif len(arrangement) > 2 and arrangement[i + 2] in vendor_neighbors:
                    index = vendor_neighbors.index(arrangement[i + 2])
                    confirmed_neighbors[index] = 0.5
            elif i == len(arrangement) - 1:
                if arrangement[i - 1] in vendor_neighbors:
                    index = vendor_neighbors.index(arrangement[i - 1])
                    confirmed_neighbors[index] = 1
                elif i > 1 and arrangement[i - 2] in vendor_neighbors:
                    index = vendor_neighbors.index(arrangement[i - 2])
                    confirmed_neighbors[index] = 0.5
            else:
                if arrangement[i - 1] in vendor_neighbors:
                    index = vendor_neighbors.index(arrangement[i - 1])
                    confirmed_neighbors[index] = 1
                elif i > 1 and arrangement[i - 2] in vendor_neighbors:
                    index = vendor_neighbors.index(arrangement[i - 2])
                    confirmed_neighbors[index] = 0.5
                if arrangement[i + 1] in vendor_neighbors:
                    index = vendor_neighbors.index(arrangement[i + 1])
                    confirmed_neighbors[index] = 1
                elif i < len(arrangement) - 2 and arrangement[i + 2] in vendor_neighbors:
                    index = vendor_neighbors.index(arrangement[i + 2])
                    confirmed_neighbors[index] = 0.5

            weights = [3, 2, 1]
            max_score = sum(weights[:len(vendor_neighbors)])
            actual_score = sum(w * c for w, c in zip(weights, confirmed_neighbors))
            success_rate = (actual_score / max_score) * 100 if max_score > 0 else 0

            for j, neighbor in enumerate(vendor_neighbors):
                if j < len(confirmed_neighbors) and confirmed_neighbors[j] > 0:
                    carer_key = f'N{j+1}_IS_CARER'
                    reader_key = f'N{j+1}_SHARE_CARD_READER'
                    if carer_key in vendors[vendor] and reader_key in vendors[vendor]:
                        if vendors[vendor][carer_key] == 'Y' and vendors[vendor][reader_key] == 'Y':
                            success_rate *= 1.2

        total_success += success_rate

        # Calculate distance penalty (positive score)
        if len(distance_vendors) == 0:
            penalty = 0
        else:
            for dist_vendor in distance_vendors:
                if dist_vendor in arrangement:
                    distance = abs(arrangement.index(dist_vendor) - i)
                    max_distance = len(arrangement) - 1  # Maximum possible distance in the list
                    if distance == 0:
                        penalty = -100  # Heavy penalty for being next to each other
                    else:
                        penalty = (distance / max_distance) * 100  # Positive score for distance

                    total_penalties += penalty / len(distance_vendors)  # Normalize penalty

        detailed_scores.append({
            "vendor": vendor,
            "confirmed_neighbors": confirmed_neighbors,
            "neighbor_success_rate": success_rate,
            "distance_penalty": total_penalties / len(arrangement) if len(distance_vendors) > 0 else 0
        })

    if num_vendors_with_requests == 0:
        overall_success = 0
        overall_penalty = 0
    else:
        overall_success = total_success / num_vendors_with_requests
        overall_penalty = total_penalties / num_vendors_with_requests

    adjusted_success = overall_success + overall_penalty  # Note addition for positive impact
    return adjusted_success, detailed_scores

# Genetic algorithm to find the best arrangement of vendors
def genetic_algorithm(vendors_df, optimal_score=80, population_size=100, generations=500):
    # Convert DataFrame to a dictionary of vendors
    vendors = vendors_df.set_index('VEND_NUMB').apply(lambda x: {
        'neighbors': [x['N_ONE'], x['N_TWO'], x['N_THREE']],
        'N1_IS_CARER': x['ONE_IS_CARER'],
        'N1_SHARE_CARD_READER': x['ONE_SHARE_CARD_READER'],
        'N2_IS_CARER': x['TWO_IS_CARER'],
        'N2_SHARE_CARD_READER': x['TWO_SHARE_CARD_READER'],
        'N3_IS_CARER': x['THREE_IS_CARER'],
        'N3_SHARE_CARD_READER': x['THREE_SHARE_CARD_READER'],
        'distance_one': x['DISTANCE_ONE'],
        'distance_two': x['DISTANCE_TWO'],
        'distance_three': x['DISTANCE_THREE']
    }, axis=1).to_dict()

    # Create an individual arrangement by shuffling the vendors
    def create_individual(vendors):
        arrangement = list(vendors.keys())
        random.shuffle(arrangement)
        return arrangement

    # Mutate an individual arrangement by swapping two vendors
    def mutate(arrangement):
        idx1, idx2 = random.sample(range(len(arrangement)), 2)
        arrangement[idx1], arrangement[idx2] = arrangement[idx2], arrangement[idx1]

    # Crossover two parent arrangements to create a child arrangement
    def crossover(parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end] = parent1[start:end]
        for item in parent2:
            if item not in child:
                child[child.index(None)] = item
        return child

    # Initialize the population with random individuals
    population = [create_individual(vendors) for _ in range(population_size)]

    best_individual = None
    best_fitness = 0
    best_detailed_scores = []

    # Iterate through generations
    for generation in tqdm(range(generations), desc="Generations"):
        fitness_scores = [(individual, calculate_success_rate(individual, vendors)) for individual in population]
        fitness_scores.sort(key=lambda x: x[1][0], reverse=True)

        if fitness_scores[0][1][0] > best_fitness:
            best_fitness = fitness_scores[0][1][0]
            best_individual = fitness_scores[0][0]
            best_detailed_scores = fitness_scores[0][1][1]

        if best_fitness >= optimal_score:
            break

        next_population = fitness_scores[:population_size // 2]
        next_population = [x[0] for x in next_population]

        while len(next_population) < population_size:
            parent1, parent2 = random.sample(next_population, 2)
            child = crossover(parent1, parent2)
            if random.random() < 0.1:
                mutate(child)
            next_population.append(child)

        population = next_population

    return best_individual, best_fitness, best_detailed_scores

# Get optimal score from user input
optimal_score_input = input("Enter optimal score (0-100, press Enter to use default 80): ")
optimal_score = int(optimal_score_input) if optimal_score_input else 80
print(f"Optimal score set to: {optimal_score}")

# The initial population size and number of generations, adjusting this will increase/decrease time taken to calculate but may result in better/alternative outputs
population_size = 100
generations = 500

# Run the genetic algorithm to find the best arrangement
best_individual, best_fitness, best_detailed_scores = genetic_algorithm(vendors_df, optimal_score=optimal_score, population_size=population_size, generations=generations)

# Print the best arrangement and its fitness score
print("Best Arrangement:", best_individual)
print("Best Fitness Score:", best_fitness)
print("Detailed Scores:")
for score in best_detailed_scores:
    print(f"Vendor: {score['vendor']}")
    print(f"  Confirmed Neighbors: {score['confirmed_neighbors']}")
    if score['neighbor_success_rate'] is not None and score['distance_penalty'] is not None:
        print(f"  Neighbor Success Rate: {score['neighbor_success_rate']}%")
        print(f"  Distance Penalty: {score['distance_penalty']}")
        print(f"  Overall Score: {score['neighbor_success_rate'] - score['distance_penalty']}%\n")
    else:
        print("  No neighbor or distance requests\n")

# Reload the best_detailed_scores dict into a pandas dataframe
best_fit_df = pd.DataFrame.from_dict(best_detailed_scores)

# Add the best fitness score in as a new column
best_fit_df['best_fitness'] = best_fitness

# Assign table numbers for visualisation and distance calculation, starting from 1
best_fit_df.insert(0, 'Table Number', range(1, 1 + len(best_fit_df)))

# Add in 3 unique columns which takes the list column and splits the data into the relevant fields
df3 = pd.DataFrame(best_fit_df['confirmed_neighbors'].to_list(), columns=['neighbour_one_confirmed', 'neighbour_two_confirmed', 'neighbour_three_confirmed'])

# Merge the dataframes together using the vendor ID PK
merged_df = best_fit_df.merge(vendors_df, left_on='vendor' , right_on='VEND_NUMB' )
final_df = merged_df.merge(df3, left_index=True, right_index=True)

# Function to get the Table Number based on DISTANCE_ONE value
def get_table_number(vendor_name, df):
    if vendor_name in df['vendor'].values:
        return df[df['vendor'] == vendor_name]['Table Number'].values[0]
    return None

# Calculate the distance between the vendor and their first distance request, return the number of tables between the source vendor and the distance vendor
final_df['DISTANCE_ONE_TABLE_LOCATION'] = final_df['DISTANCE_ONE'].apply(lambda x: get_table_number(x, final_df))
final_df['TABLE_DISTANCE_DIFFERENCE_ONE'] = (final_df['Table Number'] - final_df['DISTANCE_ONE_TABLE_LOCATION']).abs()
final_df['TABLE_DISTANCE_DIFFERENCE_ONE'] = final_df['TABLE_DISTANCE_DIFFERENCE_ONE'].fillna(0).astype(int)

# Calculate the distance between the vendor and their second distance request, return the number of tables between the source vendor and the distance vendor
final_df['DISTANCE_TWO_TABLE_LOCATION'] = final_df['DISTANCE_TWO'].apply(lambda x: get_table_number(x, final_df))
final_df['TABLE_DISTANCE_DIFFERENCE_TWO'] = (final_df['Table Number'] - final_df['DISTANCE_TWO_TABLE_LOCATION']).abs()
final_df['TABLE_DISTANCE_DIFFERENCE_TWO'] = final_df['TABLE_DISTANCE_DIFFERENCE_TWO'].fillna(0).astype(int)

# Calculate the distance between the vendor and their third distance request, return the number of tables between the source vendor and the distance vendor
final_df['DISTANCE_THREE_TABLE_LOCATION'] = final_df['DISTANCE_THREE'].apply(lambda x: get_table_number(x, final_df))
final_df['TABLE_DISTANCE_DIFFERENCE_THREE'] = (final_df['Table Number'] - final_df['DISTANCE_THREE_TABLE_LOCATION']).abs()
final_df['TABLE_DISTANCE_DIFFERENCE_THREE'] = final_df['TABLE_DISTANCE_DIFFERENCE_THREE'].fillna(0).astype(int)

# Save the final dataframe into a CSV in the root folder
final_df.to_csv(f'VENDOR_ORDER_OUTPUT.csv')

# Display the dataframe in the console
final_df