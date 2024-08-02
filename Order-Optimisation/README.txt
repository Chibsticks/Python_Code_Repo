# Order Optimisation

## Automated vendor 'table order' generation using Genetic algorithm to provide the most optimal vendor order based on neighbour preference, carer requirements, shared card reader requirements, and distance requests


### How to use:
- You will need an up to date python environment installed on your computer.
- Git Clone or download the MAIN repo or this folder directly.
- I use Jupyter Labs Notebooks for my code development, however you do not need this to run the script.
    - If you run your python environment in the native console, navigate to this folder and run the Order_Optimisation_Algorithm_v2.py script, you will need the VENDORS_CSV sample data in the same folder for the script to work.
    - If you use notebooks, you can either run the .py or the .ipynb file.
    - You will need to remember that you have the !pip install requirements installed (These are in the first cell of the ipynb file.)

## Improvements & Collaboration
If you'd like to get involved and make some suggestions or code changes please get in touch :)

## Current Issues

1. The success rate is calculating incorrectly for vendors with no neighbour requests or distance requests, which is causing an error in calculating the optimal score.

## Future Developments

1. Wall and Isle table categories and shapes.
2. Accessibility Requirements