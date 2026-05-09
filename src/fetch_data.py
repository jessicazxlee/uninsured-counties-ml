import datacommons_pandas as dc
import pandas as pd

# MOVED TO PREPROCESSING AS WELL - USE ONLY FOR TESTING
def fetch_county_dataset():

    counties = dc.get_places_in(["country/USA"], "County")
    counties = counties["country/USA"]

    variables = [
        "UninsuredRate_Person",
        "Median_Income_Household",
        "PovertyRate_Person",
        "UnemploymentRate_Person",
        "Count_Person",
        "PopulationDensity",
    ]

    df = dc.build_multivariate_dataframe(counties, variables)

    df = df.reset_index()
    df.rename(columns={"index": "county"}, inplace=True)

    return df


if __name__ == "__main__":

    df = fetch_county_dataset()

    print("Dataset shape:", df.shape)

    df.to_csv("../data/county_dataset.csv", index=False)
