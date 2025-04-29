The following people contributed to the project:
Rebecca Bubis, Yashaswi Maguluri, Brendan Wilcox

Altman Z-Score Inspired Industry-Specific Model

What, How, Why?

What?
This app predicts bankruptcy risk using financial ratios and a logistic regression model tailored by industry groupings. It draws inspiration from Altman's Z-score but is expanded for modern datasets.

Why?
Early prediction of bankruptcy risk helps prioritize audits, deeper analysis, and proactive financial decisions.

How?
- 8 core financial ratios are standardized and combined with industry dummy variables.
- These are 5 of the 10 industries we tested the model on and which performed the best: Software & Services (4510) Capital Goods (2010), Consumer Services (2530), Household & Personal Products (3030), Materials (1510)
- A logistic regression model estimates bankruptcy probability over a 1-year period.
- No subjective judgments — purely quantitative early warning system.

Model trained as of April 25, 2025.
