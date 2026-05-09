# ML_4641_Team_29

This repository contains our CS 4641 machine learning project on modeling county-level uninsured rates in the United States using socioeconomic indicators from Data Commons. The project focuses on building a preprocessing pipeline, training baseline models, and evaluating predictive performance with regression metrics and visualizations.

## Project Goal

The goal of this project is to use county-level features such as population, uninsured households, median household income, and unemployment rate to predict insurance vulnerability across U.S. counties. Our current baseline approach uses an `ElasticNetCV` regression pipeline with preprocessing for missing values, scaling, and categorical encoding.

## Repository Structure

/docs/: Contains the files used to publish the GitHub Pages site.  
/docs/index.md: Main GitHub Pages entry page that displays the midterm report content.  
/docs/figures/: Stores images and other visual assets used by the GitHub Pages site.  
/docs/figures/actual_vs_predicted.png: Scatter plot comparing actual uninsured rates to model predictions.  

/src/: Contains the main source code for data fetching, preprocessing, modeling, and evaluation.  
/src/Metrics.py: Computes regression metrics such as MAE, RMSE, and R², and generates the actual vs predicted visualization.  
/src/Models.py: Defines and trains the baseline `ElasticNetCV` regression pipeline and prints model evaluation results.  
/src/Preprocess.py: Fetches county-level data from Data Commons, saves and loads datasets, splits data, identifies feature types, and builds the preprocessing pipeline using imputers, scaling, and one-hot encoding.  
/src/fetch_data.py: Older testing script for fetching county-level data using `datacommons_pandas`; retained mainly for debugging and comparison purposes.  

/.gitignore: Specifies files and folders that Git should ignore.  
/Proposal.md: Initial project proposal describing the motivation, problem definition, planned methods, evaluation metrics, and team contributions.  
/README.md: Main project documentation describing the repository structure, project purpose, setup, and usage instructions.  
/requirements.txt: Lists the Python dependencies required to run the project.  

## Files and Code Summary

The project is currently organized around four main tasks:

1. Data collection  
   County-level data is fetched from Data Commons through the `DataCommonsClient` in `Preprocess.py`. The fetched variables currently include uninsured households, total population, median household income, and unemployment rate.

2. Preprocessing  
   The preprocessing pipeline handles missing values, scaling, and categorical encoding while preventing data leakage by splitting into training and test sets before transformation.

3. Modeling  
   The baseline model is implemented in `Models.py` using `ElasticNetCV` inside a scikit-learn `Pipeline`.

4. Evaluation  
   `Metrics.py` evaluates predictions using MAE, RMSE, and R², and creates a scatter plot of actual versus predicted uninsured rates.

## Dataset

This project uses county-level data from Data Commons.

Dataset link:  
[Data Commons – Uninsured Rates by County](https://datacommons.org/explore#client=ui_landing&q=Which+counties+in+the+US+have+the+highest+rates+of+uninsured)

## Current Pipeline

The current baseline workflow is:

- Fetch county-level data from Data Commons
- Construct `UninsuredRate` as:
  
  `Count_Household_NoHealthInsurance / Count_Person`

- Drop rows with missing target values
- Split data into training and testing sets
- Preprocess features with:
  - median imputation for numeric columns
  - most frequent imputation for categorical columns
  - standard scaling for numeric columns
  - one-hot encoding for categorical columns
- Train an `ElasticNetCV` regression model
- Evaluate performance using MAE, RMSE, and R²
- Save the actual vs predicted plot in `/figures/`

## Installation

Clone the repository and install dependencies:

```bash
git clone <your-repo-link>
cd ML_4641_Team_29
pip install -r requirements.txt
