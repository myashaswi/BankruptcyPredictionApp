The following people contributed to the project:
Rebecca Bubis, Yashaswi Maguluri, Brendan Wilcox

Altman Z-Score Inspired Industry-Specific Model
What, How, Why?
By: Rebecca Bubis, Yashaswi Maguluri, Brendan Wilcox
What?
This app predicts bankruptcy risk using financial ratios and a logistic regression model tailored by industry groupings. It draws inspiration from Altman's Z-score but is expanded for modern datasets.

Why?
Early prediction of bankruptcy risk helps prioritize audits, deeper analysis, and proactive financial decisions.

How?
8 core financial ratios are standardized and combined with industry dummy variables.

These are 5 of the 10 industries we tested the model on and which performed the best: Software & Services (4510) Capital Goods (2010), Consumer Services (2530), Household & Personal Products (3030), Materials (1510)

A logistic regression model estimates bankruptcy probability over a 1-year period.

No subjective judgments — purely quantitative early warning system.

The formula we used to predict bankruptcy is:

z
=
∑
i
=
1
10
α
i
⋅
i
n
d
u
s
t
r
y
i
+
∑
j
=
1
8
β
j
⋅
r
a
t
i
o
j
z= 
i=1
∑
10
​
 α 
i
​
 ⋅industry 
i
​
 + 
j=1
∑
8
​
 β 
j
​
 ⋅ratio 
j
​
 
The above formula when expanded is:

z
=
α
1
⋅
i
n
d
u
s
t
r
y
1
+
α
2
⋅
i
n
d
u
s
t
r
y
2
+
⋯
+
α
10
⋅
i
n
d
u
s
t
r
y
10
+
β
1
⋅
r
a
t
i
o
1
+
β
2
⋅
r
a
t
i
o
2
+
⋯
+
β
8
⋅
r
a
t
i
o
8
z=α 
1
​
 ⋅industry 
1
​
 +α 
2
​
 ⋅industry 
2
​
 +⋯+α 
10
​
 ⋅industry 
10
​
 +β 
1
​
 ⋅ratio 
1
​
 +β 
2
​
 ⋅ratio 
2
​
 +⋯+β 
8
​
 ⋅ratio 
8
​
 
Logistic Probability Formula:

P
(
Bankruptcy
=
1
∣
X
)
=
1
1
+
e
−
z
P(Bankruptcy=1∣X)= 
1+e 
−z
 
1
​
 
Model trained as of April 25, 2025.
