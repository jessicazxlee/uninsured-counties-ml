# ML_4641_Team_29
ML 4641 Project Proposal

**1. Introduction / Background**

Health insurance coverage is a critical determinant of healthcare access, preventive service utilization, and long-term health outcomes in the United States. Counties with high uninsured rates often experience poorer health outcomes, increased emergency room utilization, and greater financial strain on public health systems. Prior research has demonstrated strong associations between insurance coverage and both mortality and access to care [1], while socioeconomic determinants such as income, employment, and education significantly influence coverage disparities [2]. Additionally, geographic disparities in healthcare access remain persistent despite federal policy efforts such as the Affordable Care Act (ACA) [3].

This project will leverage county-level data from Data Commons, a publicly accessible knowledge graph aggregating U.S. Census, CDC, and other federal data sources. The primary dataset explores
“Which counties in the US have the highest rates of uninsured?”

Dataset link: [Data Commons – Uninsured Rates by County](https://datacommons.org/explore#client=ui_landing&q=Which+counties+in+the+US+have+the+highest+rates+of+uninsured)

The dataset provides county-level measures of health insurance coverage and its socioeconomic, demographic, and structural determinants. Specifically, the dataset captures:

- Insurance Coverage Metrics: Total uninsured rate and coverage breakdowns by age group.

- Demographic Composition: Population size, age distribution, racial and ethnic composition, and gender distribution.

- Socioeconomic Indicators: Median household income, poverty rate, unemployment rate, labor force participation, and educational attainment levels.

- Health and Access Proxies: Indicators related to healthcare access.

- Geographic and Structural Characteristics: Urban–rural classification, regional location, and population density.

These features enable both descriptive and predictive modeling of uninsured rates across U.S. counties. Because the dataset integrates socioeconomic and demographic indicators, it is well-suited for supervised and unsupervised machine learning approaches to uncover structural patterns and predictive relationships.

Existing literature has primarily focused on national or state-level trends [1][3], with fewer predictive modeling studies at the county level. Machine learning offers an opportunity to move beyond correlation toward identifying nonlinear relationships and clustering counties with similar risk profiles.

**2. Problem Definition**

Problem:

Can we use county-level socioeconomic and demographic features to predict and identify high-risk counties with elevated uninsured rates, and uncover structural clusters of counties with similar insurance vulnerability profiles?

We aim to predict uninsured rate as a continuous variable, classify counties into high-risk vs. low-risk categories, and identify latent clusters of counties with similar socioeconomic patterns.


Motivation:

Healthcare access inequities remain a pressing national concern. Counties with persistently high uninsured rates often overlap with economically disadvantaged or rural regions. However, policymakers typically rely on descriptive statistics rather than predictive tools.

A machine learning framework could enable early identification of at-risk counties, support data-driven policy allocation of healthcare resources, and reveal nonlinear interactions between poverty, employment, education, and insurance coverage.

Beyond predictive performance, this project contributes to sustainability and ethical governance by promoting equitable healthcare access. 

**3. Methods**

We will build a county-level tabular ML pipeline in scikit-learn using Data Commons features (insurance, demographic, socioeconomic, and geographic indicators) to model uninsured rates.

Preprocessing Methods:
- Impute missing values (SimpleImputer): median for skewed numerics (e.g., income, density) and most-frequent for categoricals—fits county data where fields are often incomplete.
- Encode categoricals (OneHotEncoder, handle_unknown="ignore"): for nominal fields like region and urban/rural.
- Scale features (StandardScaler): helps linear/logistic models since variables vary widely in scale.
- Prevent leakage (ColumnTransformer + Pipeline): keeps all preprocessing inside cross-validation and applies the right transforms per column.

Models:
- ElasticNetCV (baseline regression): interpretable, handles correlated predictors, regularizes to reduce overfitting.
- RandomForestRegressor: captures nonlinearities and interactions; strong, robust tabular baseline.
- HistGradientBoostingRegressor: often improves MAE/RMSE on structured data via more complex patterns.

Classification (high-risk vs low-risk):
- LogisticRegression (baseline): fast, interpretable, policy-friendly.
- RandomForestClassifier: nonlinear boundaries + interactions; can improve recall/F1 for vulnerable counties.

We will use cross-validation (e.g., GridSearchCV) to tune and compare models fairly and improve generalization.

**4. Results and Discussion**

Several quantitative metrics guide the project. Given the dataset’s sensitive socioeconomic attributes, fairness, transparency, and responsible interpretation are essential. Improving identification of counties with elevated uninsured rates supports equitable healthcare resource allocation.

The first objective is predicting county-level uninsured rates as a continuous outcome. Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) will measure prediction error, while the Coefficient of Determination (R²) will assess explained variance. The goal is to minimize MAE and RMSE and achieve strong explanatory power, indicating that socioeconomic and demographic features account for variation across counties.

The second objective is classifying counties as high or low risk. Performance will be evaluated using accuracy, precision, recall, F1-score, and ROC-AUC. Recall and F1-score are prioritized to avoid overlooking high-risk counties while maintaining balanced precision. A strong ROC-AUC would indicate effective discrimination across thresholds, which could outperform linear classifiers due to nonlinear feature interactions.

Finally, clustering will identify counties with similar socioeconomic profiles. Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score will assess cohesion and separation. The goal is well-separated, internally consistent clusters, reflected by higher Silhouette and Calinski-Harabasz scores and lower Davies-Bouldin values, which could reveal regional or rural-urban vulnerability patterns [4].


**5. References**

[1] B. D. Sommers, A. E. Gawande, and K. Baicker, “Health Insurance Coverage and Health — What the Recent Evidence Tells Us,” New England Journal of Medicine, vol. 377, no. 6, pp. 586–593, 2017.
https://www.nejm.org/doi/full/10.1056/NEJMsb1706645

[2] A. S. Wilper et al., “Health Insurance and Mortality in US Adults,” American Journal of Public Health, vol. 99, no. 12, pp. 2289–2295, 2009.
https://ajph.aphapublications.org/doi/10.2105/AJPH.2008.157685

[3] K. Baicker et al., “The Oregon Experiment — Effects of Medicaid on Clinical Outcomes,” New England Journal of Medicine, vol. 368, pp. 1713–1722, 2013.
https://www.nejm.org/doi/full/10.1056/NEJMsa1212321

[4] scikit-learn developers, “Model evaluation: Quantifying the quality of predictions,” scikit-learn, 2025. [Online]. Available: https://scikit-learn.org/stable/modules/model_evaluation.html. [Accessed: 24-Feb-2026].

**Contribution Table**

| Name | Proposal Contribution |
|---------|----------|
| Aayush P | Research on dataset, Section 3. Methods|
| Jessica L | Slideshow for Video Proposal, Recording/Editing Video Proposal |
| Mona B | Sections 1. Introduction, Section 2. Problem Definition |
| Jelena C | Sections 4. Results and Discussion |
| Daanish M | Gantt Chart |

**Video Presentation**

<https://youtu.be/dEP_7qmkjAY>

**Gantt Chart**

<https://docs.google.com/spreadsheets/d/1haZeVYoj3DfT7yv5yEm-OdRzf2P-u45e/edit?usp=sharing&ouid=107279860147909644308&rtpof=true&sd=true>
