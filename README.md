## Code base for the publication: Winzell, Filip, et al. "Benchmarking privacy and utility in synthetic tabular cohorts for Alzheimer's disease research." Alzheimer's & Dementia: Diagnosis, Assessment & Disease Monitoring 18.3 (2026)

Pyhton code was used for synthetic data generation and utility and privacy evaluation

R code was used for creating the real datasets, and generating the plots of the paper.

If you have any questions about the repo - feel free to contact me at filip.winzell@math.lth.se or filip.winzell@di.ku.dk

**Abstract**
INTRODUCTION
The scarcity of large, clinically relevant cohorts is becoming a bottleneck in Alzheimer's disease (AD) research, as their sensitive nature makes open data sharing difficult. Privacy-preserving synthetic datasets generated with machine learning may help address this challenge.

METHODS
We compared five frameworks for generating synthetic tabular data from the Alzheimer's Disease Neuroimaging Initiative and Anti-Amyloid Treatment in Asymptomatic Alzheimer's Disease cohorts, with a set of empirical privacy and utility metrics. Two of the methods, DataSynthesizer and TableDiffusion, provide $\epsilon$-differential privacy guarantees.

RESULTS
Methods with differential privacy achieved high privacy ratings but low levels of utility. Deep learning methods like Tabular Prior-data Fitted Network (TabPFN) and Conditional Generative Adversarial Network (CTGAN) also showed high privacy with limited utility. In contrast, non-private DataSynthesizer and Synthpop offered higher utility at a cost of lower privacy.

DISCUSSION
The evaluated methods demonstrated a clear trade-off between privacy and utility. High privacy was generally associated with insufficient utility, highlighting the need for further research into synthetic data generation for AD.


