# %% [markdown]
# # Telco Customer Churn - Exploratory Data Analysis
#
# This analysis identifies customer characteristics, service patterns and financial factors associated with churn. The findings support
# preprocessing, model development and customer retention decisions.
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

DATA_DIR = Path.cwd() / "data"
# Separate outputs allow side-by-side validation without overwriting the original.
OUTPUT_DIR = DATA_DIR / "out" / "business_eda_refactored"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:,.2f}".format)

# %% [markdown]
# ## 00. Configuration
#
# Consistent visual notation and reusable functions support reproducible analysis and IBCS oriented business reporting.
# %%
IBCS_ACTUAL = "#333333"
IBCS_POSITIVE = "#008000"
IBCS_NEGATIVE = "#C00000"
sns.set_theme(style="white", context="notebook")
plt.rcParams.update(
    {
        "figure.figsize": (10, 5),
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.grid": False,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


# %%
def ibcs_bar_chart(
    data,
    category,
    value,
    message,
    measure,
    unit="%",
    filename=None,
    xmax=None,
    decimals=1,
    sort=True,
    benchmark=None,
):
    df = data[[category, value]].dropna()
    if sort:
        df = df.sort_values(value, ascending=False)
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.45 + 2.5)))
    bars = ax.barh(df[category].astype(str), df[value], color=IBCS_ACTUAL, height=0.6)
    ax.bar_label(
        bars, labels=[f"{v:,.{decimals}f}{unit}" for v in df[value]], padding=5
    )
    if benchmark is not None:
        ax.axvline(benchmark, color="#808080", linestyle="--", linewidth=1)
        ax.text(
            benchmark,
            1.02,
            f"Overall: {benchmark:.1f}%",
            transform=ax.get_xaxis_transform(),
            ha="center",
            fontsize=9,
        )
    ax.set_xlim(0, xmax if xmax is not None else max(df[value].max() * 1.2, 1))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(True)
    fig.text(0.02, 0.96, message, weight="bold", size=14, va="top")
    fig.text(0.02, 0.87, "Telco Customer Retention")
    fig.text(0.02, 0.83, measure, weight="bold")
    fig.text(0.02, 0.79, "Q3 | AC")
    fig.text(0.02, 0.02, "Source: Telco Customer Churn dataset.", size=8)
    fig.subplots_adjust(left=0.28, right=0.90, top=0.70, bottom=0.13)
    if filename:
        fig.savefig(FIGURES_DIR / filename)
    return fig, ax


# %%
def segment_summary(df, feature):
    summary = (
        df.groupby(feature, observed=True, dropna=False)
        .agg(
            Customers=("Churn", "size"),
            Churned=("Churn", "sum"),
            Churn_Rate=("Churn", "mean"),
        )
        .reset_index()
    )
    summary["Lift"] = summary["Churn_Rate"] / df["Churn"].mean()
    summary["Churn_Rate"] *= 100
    summary["Churn_Share"] = summary["Churned"] / df["Churn"].sum() * 100
    return summary


# %%
def print_summaries(results):
    for name, summary in results.items():
        print(f"\n{name}")
        print(summary.to_string(index=False))


def export_segments(results, filename):
    combined = pd.concat(
        [
            summary.rename(columns={feature: "Segment"}).assign(Feature=feature)
            for feature, summary in results.items()
        ],
        ignore_index=True,
    )
    combined.to_csv(OUTPUT_DIR / filename, index=False)


def count_summary(values, feature):
    summary = (
        values.value_counts(dropna=False)
        .rename_axis(feature)
        .reset_index(name="Customers")
    )
    summary["Share (%)"] = summary["Customers"] / len(values) * 100
    return summary


def summarize_features(df, features):
    results = {feature: segment_summary(df, feature) for feature in features}
    print_summaries(results)
    return results


# %% [markdown]
# ## 01. Data Understanding & Quality
#
# We validate customer-level integrity and compare the integrated data with the original sources to identify preprocessing issues.
# %%
telco = pd.read_excel(DATA_DIR / "out" / "telco.xlsx")
status = pd.read_excel(DATA_DIR / "Telco_customer_churn_status.xlsx")
services = pd.read_excel(
    DATA_DIR / "Telco_customer_churn_services.xlsx", keep_default_na=False
)
for name, df in {"Telco": telco, "Status": status, "Services": services}.items():
    assert df["Customer ID"].notna().all()
    assert df["Customer ID"].is_unique
    print(f"{name}: {df.shape}, unique customer IDs")
assert set(telco["Customer ID"]) == set(status["Customer ID"])
assert set(telco["Customer ID"]) == set(services["Customer ID"])
print("Customer integrity checks passed.")
# %%
quality = pd.DataFrame(
    {
        "Type": telco.dtypes.astype(str),
        "Unique": telco.nunique(),
        "Missing": telco.isna().sum(),
        "Missing (%)": telco.isna().mean() * 100,
    }
)
print(quality)
missing = quality.query("Missing > 0").reset_index(names="Feature")
fields = ["Offer", "Internet Type"]
validation = pd.DataFrame(
    {
        "Integrated Missing": telco[fields].isna().sum(),
        "Original None": services[fields].eq("None").sum(),
        "Original Empty": services[fields].eq("").sum(),
    }
)
print("\nMissing value validation:")
print(validation)
quality.to_csv(OUTPUT_DIR / "data_quality.csv")
validation.to_csv(OUTPUT_DIR / "categorical_validation.csv")
# %%
if not missing.empty:
    ibcs_bar_chart(
        missing,
        "Feature",
        "Missing (%)",
        message="Missing values are concentrated in two service features.",
        measure="Missing values by feature, %",
        filename="01_missing_values.png",
        xmax=100,
    )
    plt.show()
# %% [markdown]
# ### Key Insights
#
# The integrated dataset contains 7,043 unique customers and 41 variables, with no duplicate customer IDs. Missing values in Offer (55.05%) and Internet Type (21.67%) match the counts of literal "None" values in the original source, suggesting
# an import-related issue rather than missing customer information.

# %% [markdown]
# ## 02. Target Variable Analysis
#
# We define the eligible customer population and encode the target variable to establish a consistent churn baseline.
# Newly joined customers are excluded from the retention analysis.
# %%
eda = telco.merge(
    status[["Customer ID", "Customer Status"]], on="Customer ID", validate="one_to_one"
)
print("Original customer status:")
print(eda["Customer Status"].value_counts())
eda = eda.loc[eda["Customer Status"].isin(["Stayed", "Churned"])].copy()
# Restore original categories for descriptive analysis only.
source = services.set_index("Customer ID")
for feature in ["Offer", "Internet Type"]:
    eda[feature] = eda["Customer ID"].map(source[feature])
eda["Churn"] = eda["Churn Label"].map({"Yes": 1, "No": 0})
assert eda["Churn"].notna().all()
assert eda["Churn"].eq(eda["Customer Status"].eq("Churned").astype(int)).all()
CHURN_BASELINE = eda["Churn"].mean() * 100
print(f"\nEligible customers: {len(eda):,}")
print(f"Overall churn rate: {CHURN_BASELINE:.2f}%")
# %%
target_summary = (
    eda["Customer Status"]
    .value_counts()
    .rename_axis("Status")
    .reset_index(name="Customers")
)
target_summary["Share (%)"] = target_summary["Customers"] / len(eda) * 100
print(target_summary.to_string(index=False))
ibcs_bar_chart(
    target_summary,
    "Status",
    "Share (%)",
    message=f"{CHURN_BASELINE:.1f}% of eligible customers have churned.",
    measure="Customer status distribution, %",
    filename="02_target_distribution.png",
    xmax=100,
)
plt.show()
# %% [markdown]
# ### Key Insights
#
# The eligible population contains 6,589 customers, with an observed churn rate of 28.37%. The moderate class imbalance means that accuracy alone is insufficient to evaluate churn prediction models.

# %% [markdown]
# ## 03. Customer Demographics & Geography
#
# We compare churn across demographic and geographic segments to identify differences in customer retention and potential sources of heterogeneity.
# %%
eda["Age Group"] = pd.cut(
    eda["Age"],
    bins=[0, 29, 44, 64, np.inf],
    labels=["Under 30", "30-44", "45-64", "65+"],
)
demographic_features = ["Gender", "Age Group", "Married", "Dependents"]
demographic_results = summarize_features(eda, demographic_features)
# %%
for feature in ["Age Group", "Dependents"]:
    summary = demographic_results[feature]
    highest = summary.loc[summary["Churn_Rate"].idxmax()]
    ibcs_bar_chart(
        summary,
        feature,
        "Churn_Rate",
        message=(
            f"{highest[feature]} has the highest observed "
            f"churn rate ({highest['Churn_Rate']:.1f}%)."
        ),
        measure=f"Churn rate by {feature.lower()}, %",
        benchmark=CHURN_BASELINE,
        xmax=100,
        sort=feature != "Age Group",
        filename=f"03_{feature.lower().replace(' ', '_')}.png",
    )
plt.show()
# %% [markdown]
# ### Geographic Distribution
#
# We compare churn rates across cities with at least 50 customers to reduce the influence of small and potentially unstable samples.
# %%
city_summary = segment_summary(eda, "City")
city_summary = city_summary.loc[city_summary["Customers"] >= 50].sort_values(
    "Churn_Rate", ascending=False
)
print(city_summary.to_string(index=False))
ibcs_bar_chart(
    city_summary.head(10),
    "City",
    "Churn_Rate",
    message="Observed churn rates vary across major customer locations.",
    measure="Churn rate by city, %",
    benchmark=CHURN_BASELINE,
    xmax=100,
    filename="03_geographic_churn.png",
)
plt.show()
# %%
export_segments(demographic_results, "demographic_analysis.csv")
city_summary.to_csv(OUTPUT_DIR / "geographic_analysis.csv", index=False)
# %% [markdown]
# ### Key Insights
#
# Gender shows limited descriptive differences, while customers aged 65+ (42.8%) and those without dependents (35.0%) show higher churn.
# San Diego records a 66.5% churn rate and accounts for 9.9% of all observed churn cases, warranting further investigation.

# %% [markdown]
# ## 04. Customer Behaviour & Services
#
# We analyze customer tenure, contracts, offers and service adoption to identify behavioral patterns associated with churn and potential
# opportunities for customer retention.
# %%
eda["Tenure Group"] = pd.cut(
    eda["Tenure in Months"],
    bins=[0, 6, 12, 24, 48, np.inf],
    labels=["1-6m", "7-12m", "13-24m", "25-48m", "49+m"],
)
behavior_features = [
    "Tenure Group",
    "Contract",
    "Offer",
    "Internet Type",
    "Payment Method",
    "Referred a Friend",
    "Phone Service",
    "Paperless Billing",
]
behavior_results = summarize_features(eda, behavior_features)
# %% [markdown]
# ### Internet Service Adoption
#
# We restrict internet-related comparisons to customers with internet access to distinguish service non-adoption from service ineligibility.
# %%
internet_customers = eda.loc[eda["Internet Service"].eq("Yes")].copy()
internet_features = [
    "Premium Tech Support",
    "Online Security",
    "Online Backup",
    "Device Protection Plan",
    "Streaming TV",
    "Streaming Movies",
    "Streaming Music",
    "Unlimited Data",
]
internet_results = summarize_features(internet_customers, internet_features)
INTERNET_BASELINE = internet_customers["Churn"].mean() * 100
print(f"\nInternet customers: {len(internet_customers):,}")
print(f"Internet churn baseline: {INTERNET_BASELINE:.2f}%")
# %%
for feature in ["Tenure Group", "Contract", "Offer", "Internet Type"]:
    summary = behavior_results[feature]
    highest = summary.loc[summary["Churn_Rate"].idxmax()]
    ibcs_bar_chart(
        summary,
        feature,
        "Churn_Rate",
        message=(
            f"{feature}: {highest[feature]} records the highest "
            f"observed churn rate ({highest['Churn_Rate']:.1f}%)."
        ),
        measure=f"Churn rate by {feature.lower()}, %",
        benchmark=CHURN_BASELINE,
        xmax=100,
        sort=feature != "Tenure Group",
        filename=f"04_{feature.lower().replace(' ', '_')}.png",
    )
for feature in ["Premium Tech Support", "Online Security"]:
    summary = internet_results[feature]
    rates = summary.set_index(feature)["Churn_Rate"]
    ibcs_bar_chart(
        summary,
        feature,
        "Churn_Rate",
        message=(
            f"{feature}: churn is {rates['Yes']:.1f}% with "
            f"the service and {rates['No']:.1f}% without."
        ),
        measure=f"Internet customers - churn by {feature.lower()}, %",
        benchmark=INTERNET_BASELINE,
        xmax=100,
        filename=f"04_{feature.lower().replace(' ', '_')}.png",
    )
plt.show()
# %%
for name, results in {
    "behavior_analysis": behavior_results,
    "internet_services_analysis": internet_results,
}.items():
    export_segments(results, f"{name}.csv")
# %% [markdown]
# ### Key Insights
#
# Churn is concentrated among customers with short tenure (77.2%) and monthly contracts, which account for 88.6% of all churn cases.
# Offer E (67.6%) and fiber optic services (42.1%) show high churn rates.
# Technical support and online security are associated with substantially lower churn among internet customers, suggesting retention hypotheses
# for further investigation.

# %% [markdown]
# ## 05. Financial Analysis
#
# We analyze customer charges and historical revenue to assess the financial exposure associated with churn and identify economically relevant customer segments.
# %%
financial_profile = eda.groupby("Customer Status").agg(
    Customers=("Customer ID", "size"),
    Avg_Monthly_Charge=("Monthly Charge", "mean"),
    Median_Monthly_Charge=("Monthly Charge", "median"),
    Avg_Total_Revenue=("Total Revenue", "mean"),
    Median_Total_Revenue=("Total Revenue", "median"),
    Total_Revenue=("Total Revenue", "sum"),
    Avg_Tenure=("Tenure in Months", "mean"),
)
print(financial_profile.T)
# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, feature in zip(axes, ["Monthly Charge", "Total Revenue"]):
    sns.boxplot(
        data=eda,
        x="Customer Status",
        y=feature,
        order=["Stayed", "Churned"],
        color=IBCS_ACTUAL,
        ax=ax,
    )
    ax.set_title(feature)
    ax.set_xlabel("")
    ax.set_ylabel("USD")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "05_financial_distributions.png")
plt.show()
# %% [markdown]
# ### Pricing and Churn
#
# We compare churn across monthly charge bands to investigate whether higher prices are associated with higher churn rates and identify potential non-linear relationships.
# %%
eda["Price Group"] = pd.cut(
    eda["Monthly Charge"],
    bins=[-np.inf, 35, 65, 95, np.inf],
    labels=["Up to $35", "$35-65", "$65-95", "Above $95"],
)
price_summary = segment_summary(eda, "Price Group")
print(price_summary.to_string(index=False))
highest = price_summary.loc[price_summary["Churn_Rate"].idxmax()]
ibcs_bar_chart(
    price_summary,
    "Price Group",
    "Churn_Rate",
    message=(
        f"The {highest['Price Group']} price group shows "
        f"the highest churn rate ({highest['Churn_Rate']:.1f}%)."
    ),
    measure="Churn rate by monthly charge, %",
    benchmark=CHURN_BASELINE,
    xmax=100,
    sort=False,
    filename="05_price_group.png",
)
plt.show()
# %% [markdown]
# ### Observed Financial Exposure
#
# We quantify the monthly charges associated with churned customers to assess historical financial exposure. This measure is not
# a forecast of future revenue losses.
# %%
churned_customers = eda.loc[eda["Churn"].eq(1)]
total_monthly_charges = eda["Monthly Charge"].sum()
observed_exposure = churned_customers["Monthly Charge"].sum()
exposure_share = observed_exposure / total_monthly_charges * 100
print(f"Total monthly charges: ${total_monthly_charges:,.2f}")
print(f"Observed churn exposure: ${observed_exposure:,.2f}")
print(f"Financial exposure share: {exposure_share:.2f}%")
print(f"Customer churn share: {CHURN_BASELINE:.2f}%")
# %%
financial_results = {}
for feature in ["Contract", "Tenure Group", "Offer", "Internet Type"]:
    summary = segment_summary(eda, feature)
    exposure = (
        churned_customers.groupby(feature, observed=True)["Monthly Charge"]
        .sum()
        .rename("Monthly_Exposure")
        .reset_index()
    )
    summary = summary.merge(exposure, on=feature, how="left")
    summary["Monthly_Exposure"] = summary["Monthly_Exposure"].fillna(0)
    summary["Exposure_Share"] = summary["Monthly_Exposure"] / observed_exposure * 100
    financial_results[feature] = summary
    print(f"\n{feature}")
    print(summary.to_string(index=False))
# %%
for feature in ["Contract", "Internet Type"]:
    summary = financial_results[feature]
    highest = summary.loc[summary["Exposure_Share"].idxmax()]
    ibcs_bar_chart(
        summary,
        feature,
        "Exposure_Share",
        message=(
            f"{highest[feature]} accounts for "
            f"{highest['Exposure_Share']:.1f}% "
            "of observed financial exposure."
        ),
        measure=f"Monthly charge exposure by {feature.lower()}, %",
        xmax=100,
        filename=f"05_exposure_{feature.lower().replace(' ', '_')}.png",
    )
plt.show()
# %%
financial_profile.to_csv(OUTPUT_DIR / "financial_profile.csv")
price_summary.to_csv(OUTPUT_DIR / "price_analysis.csv", index=False)
export_segments(financial_results, "financial_exposure.csv")
# %% [markdown]
# ### Key Insights
#
# Churned customers have 18.2% higher average monthly charges, accounting for 31.9% of total observed monthly charges.
# Month-to-month contracts and fiber optic customers concentrate 86.9% and 78.2% of observed financial exposure, respectively.
# The $65-95 price group shows the highest churn rate (37.5%), suggesting a non-linear association between pricing and churn.

# %% [markdown]
# ## 06. Statistical & Multivariate Analysis
#
# We assess the strength of associations between customer features and churn using statistical measures and feature interactions.
# The results support feature assessment and business hypotheses.
# %%
numeric_features = [
    "Age",
    "Number of Dependents",
    "Number of Referrals",
    "Tenure in Months",
    "Monthly Charge",
    "Total Charges",
    "Total Refunds",
    "Total Revenue",
    "Avg Monthly Long Distance Charges",
    "Total Long Distance Charges",
    "Avg Monthly GB Download",
    "Total Extra Data Charges",
    "Population",
]
numeric_associations = pd.DataFrame(
    {
        "Pearson": eda[numeric_features].corrwith(eda["Churn"], method="pearson"),
        "Spearman": eda[numeric_features].corrwith(eda["Churn"], method="spearman"),
    }
)
numeric_associations["Abs_Pearson"] = numeric_associations["Pearson"].abs()
numeric_associations = numeric_associations.sort_values("Abs_Pearson", ascending=False)
print(numeric_associations)
# %%
correlation_matrix = eda[numeric_features + ["Churn"]].corr(method="pearson")
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    annot_kws={"size": 8},
    ax=ax,
)
ax.set_title("Numerical Feature Correlations - Pearson")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "06_correlation_matrix.png")
plt.show()
# %% [markdown]
# ### Categorical Associations
#
# We use Chi-square tests and Cramer's V to assess categorical associations with churn, distinguishing statistical significance
# from association strength.


# %%
def categorical_association(df, feature):
    table = pd.crosstab(df[feature], df["Churn"])
    chi2, p_value, _, expected = stats.chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    dimensions = min(table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * dimensions))
    return {
        "Feature": feature,
        "Chi2": chi2,
        "P_Value": p_value,
        "Cramers_V": cramers_v,
        "Min_Expected_Count": expected.min(),
        "Sample_Size": n,
    }


categorical_associations = pd.DataFrame(
    [
        categorical_association(eda, feature)
        for feature in demographic_features + behavior_features
    ]
    + [
        categorical_association(internet_customers, feature)
        for feature in internet_features
    ]
).sort_values("Cramers_V", ascending=False)
print(
    categorical_associations.to_string(index=False, float_format=lambda x: f"{x:.4g}")
)
# %%
ibcs_bar_chart(
    categorical_associations.head(10),
    "Feature",
    "Cramers_V",
    message="Top categorical associations with observed churn.",
    measure="Categorical association strength, Cramer's V",
    unit="",
    decimals=3,
    xmax=1,
    filename="06_categorical_associations.png",
)
plt.show()
# %% [markdown]
# ### Numerical Group Comparisons
#
# We compare numerical distributions between stayed and churned customers using Mann-Whitney U tests and rank-biserial effect sizes.
# %%
group_tests = []
for feature in ["Tenure in Months", "Monthly Charge", "Total Revenue"]:
    churned = eda.loc[eda["Churn"].eq(1), feature].dropna()
    stayed = eda.loc[eda["Churn"].eq(0), feature].dropna()
    u_stat, p_value = stats.mannwhitneyu(churned, stayed, alternative="two-sided")
    effect_size = 2 * u_stat / (len(churned) * len(stayed)) - 1
    group_tests.append(
        {
            "Feature": feature,
            "Median_Churned": churned.median(),
            "Median_Stayed": stayed.median(),
            "U_Statistic": u_stat,
            "P_Value": p_value,
            "Rank_Biserial": effect_size,
        }
    )
group_tests = pd.DataFrame(group_tests)
print(group_tests.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
# %%
p_values = pd.concat(
    [categorical_associations["P_Value"], group_tests["P_Value"]], ignore_index=True
).to_numpy()
order = np.argsort(p_values)
adjusted = np.empty_like(p_values)
adjusted[order] = np.minimum(
    1,
    np.maximum.accumulate(p_values[order] * (len(p_values) - np.arange(len(p_values)))),
)
n_categorical = len(categorical_associations)
categorical_associations["Holm_P"] = adjusted[:n_categorical]
group_tests["Holm_P"] = adjusted[n_categorical:]
print(
    categorical_associations[["Feature", "Cramers_V", "P_Value", "Holm_P"]].to_string(
        index=False, float_format=lambda x: f"{x:.4g}"
    )
)
print(
    group_tests[["Feature", "Rank_Biserial", "P_Value", "Holm_P"]].to_string(
        index=False, float_format=lambda x: f"{x:.4g}"
    )
)
# %% [markdown]
# ### Feature Interactions
#
# We explore selected feature combinations to investigate whether observed churn patterns persist across customer segments and identify potential confounding factors.
# %%
interactions = [
    ("Contract", "Tenure Group"),
    ("Offer", "Tenure Group"),
    ("Offer", "Contract"),
    ("Internet Type", "Price Group"),
    ("Contract", "Internet Type"),
]
interaction_results = {
    f"{first} x {second}": segment_summary(eda, [first, second])
    for first, second in interactions
}
print_summaries(interaction_results)
# %%
for first, second in interactions:
    summary = interaction_results[f"{first} x {second}"]
    rates = summary.pivot(index=first, columns=second, values="Churn_Rate")
    counts = summary.pivot(index=first, columns=second, values="Customers")
    labels = (
        rates.round(1).astype(str) + "%\nn=" + counts.fillna(0).astype(int).astype(str)
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(
        rates,
        annot=labels,
        fmt="",
        cmap="Greys",
        vmin=0,
        vmax=100,
        mask=counts.lt(30) | counts.isna(),
        linewidths=0.5,
        cbar_kws={"label": "Churn rate (%)"},
        ax=ax,
    )
    ax.set_title(f"Churn Rate - {first} x {second}")
    fig.tight_layout()
    filename = f"06_{first}_{second}".lower().replace(" ", "_")
    fig.savefig(FIGURES_DIR / f"{filename}.png")
plt.show()
# %%
numeric_associations.to_csv(OUTPUT_DIR / "numerical_associations.csv")
categorical_associations.to_csv(
    OUTPUT_DIR / "categorical_associations.csv", index=False
)
group_tests.to_csv(OUTPUT_DIR / "statistical_tests.csv", index=False)
pd.concat(
    [summary.assign(Interaction=name) for name, summary in interaction_results.items()],
    ignore_index=True,
).to_csv(OUTPUT_DIR / "interaction_analysis.csv", index=False)
# %% [markdown]
# ### Key Insights
#
# Contract and tenure show the strongest categorical associations with churn (Cramer's V of 0.507 and 0.500, respectively).
# Stratified analysis reveals that offer and pricing patterns depend on customer tenure and internet technology.
# Month-to-month fiber optic customers account for 59.2% of all observed churn cases, identifying a potential retention priority.

# %% [markdown]
# ## 07. Customer Segmentation & Retention Insights
#
# We combine churn concentration, association strength and financial exposure to identify potential retention priorities. Reported churn
# reasons provide additional context for business recommendations.
# %%
monthly = eda["Contract"].eq("Month-to-Month")
fiber = eda["Internet Type"].eq("Fiber Optic")
internet = eda["Internet Service"].eq("Yes")
segments = {
    "Monthly + Early Tenure": monthly & eda["Tenure Group"].eq("1-6m"),
    "Monthly + Fiber Optic": monthly & fiber,
    "Monthly + Offer E": monthly & eda["Offer"].eq("Offer E"),
    "Monthly + No Offer": monthly & eda["Offer"].eq("None"),
    "Fiber + $65-95": fiber & eda["Price Group"].eq("$65-95"),
    "Internet + No Tech Support": internet & eda["Premium Tech Support"].eq("No"),
}
retention_results = []
for name, mask in segments.items():
    customers = eda.loc[mask]
    lost = customers.loc[customers["Churn"].eq(1)]
    rate = customers["Churn"].mean() * 100
    exposure = lost["Monthly Charge"].sum()
    retention_results.append(
        {
            "Segment": name,
            "Customers": len(customers),
            "Churned": len(lost),
            "Customer_Share": len(customers) / len(eda) * 100,
            "Churn_Rate": rate,
            "Lift": rate / CHURN_BASELINE,
            "Churn_Share": len(lost) / eda["Churn"].sum() * 100,
            "Monthly_Exposure": exposure,
            "Exposure_Share": exposure / observed_exposure * 100,
        }
    )
retention_summary = pd.DataFrame(retention_results).sort_values(
    "Monthly_Exposure", ascending=False
)
print(retention_summary.to_string(index=False))
# %%
for metric, measure, filename in [
    (
        "Churn_Rate",
        "Observed churn rate by selected segment, %",
        "07_segment_churn.png",
    ),
    (
        "Exposure_Share",
        "Observed financial exposure by selected segment, %",
        "07_segment_exposure.png",
    ),
]:
    highest = retention_summary.loc[retention_summary[metric].idxmax()]
    ibcs_bar_chart(
        retention_summary,
        "Segment",
        metric,
        message=(f"{highest['Segment']} ranks highest ({highest[metric]:.1f}%)."),
        measure=measure,
        benchmark=(CHURN_BASELINE if metric == "Churn_Rate" else None),
        xmax=100,
        filename=filename,
    )
plt.show()
# %% [markdown]
# ### Retention Priority Matrix
#
# We combine segment size, churn rate and financial exposure to compare potential retention priorities across customer groups.
# %%
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(
    retention_summary["Customers"],
    retention_summary["Churn_Rate"],
    s=retention_summary["Monthly_Exposure"] / 80,
    color=IBCS_ACTUAL,
    alpha=0.45,
    edgecolor=IBCS_ACTUAL,
)
for _, row in retention_summary.iterrows():
    ax.annotate(
        row["Segment"],
        (row["Customers"], row["Churn_Rate"]),
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=9,
    )
ax.axhline(
    CHURN_BASELINE,
    linestyle="--",
    color="grey",
    label=f"Overall churn: {CHURN_BASELINE:.1f}%",
)
ax.set(
    xlabel="Number of customers",
    ylabel="Churn rate (%)",
    ylim=(0, 100),
    title="Retention Priority Matrix",
)
ax.legend()
fig.text(
    0.02,
    0.01,
    "Bubble area represents observed monthly charge exposure. Segments may overlap.",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIGURES_DIR / "07_retention_priority.png")
plt.show()
# %% [markdown]
# ### Reported Churn Reasons
#
# We examine reported churn reasons to identify customer concerns and potential retention actions. These post-churn variables are
# used exclusively for retrospective analysis.
# %%
churn_reasons = status.loc[
    status["Customer Status"].eq("Churned"),
    ["Customer ID", "Churn Category", "Churn Reason"],
].copy()
reason_results = {}
for feature in ["Churn Category", "Churn Reason"]:
    summary = count_summary(churn_reasons[feature], feature)
    reason_results[feature] = summary
    print(f"\n{feature}")
    print(summary.to_string(index=False))
# %%
for feature in ["Churn Category", "Churn Reason"]:
    summary = reason_results[feature].head(10)
    highest = summary.iloc[0]
    ibcs_bar_chart(
        summary,
        feature,
        "Share (%)",
        message=(
            f"{highest[feature]} accounts for "
            f"{highest['Share (%)']:.1f}% of churn cases."
        ),
        measure=f"Reported {feature.lower()}, %",
        xmax=100,
        filename=f"07_{feature.lower().replace(' ', '_')}.png",
    )
plt.show()
# %%
segment_reasons = []
for name, mask in segments.items():
    ids = eda.loc[mask & eda["Churn"].eq(1), "Customer ID"]
    categories = churn_reasons.loc[
        churn_reasons["Customer ID"].isin(ids), "Churn Category"
    ]
    summary = count_summary(categories, "Churn Category")
    segment_reasons.append(summary.assign(Segment=name))
segment_reason_summary = pd.concat(segment_reasons, ignore_index=True)
print(segment_reason_summary.to_string(index=False))
# %%
retention_summary.to_csv(OUTPUT_DIR / "retention_segments.csv", index=False)
for feature, summary in reason_results.items():
    filename = feature.lower().replace(" ", "_")
    summary.to_csv(OUTPUT_DIR / f"{filename}.csv", index=False)
segment_reason_summary.to_csv(OUTPUT_DIR / "segment_churn_reasons.csv", index=False)
# %% [markdown]
# ### Key Insights
#
# Monthly customers with early tenure show the highest observed churn rate (81.3%), while internet customers without premium technical support account for 79.6% of observed financial exposure.
# Competition accounts for 45.0% of reported churn cases.
# Retention initiatives should investigate early customer engagement, contract incentives, service quality and competitive positioning.

# %% [markdown]
# ## 08. Conclusions & Key Findings
#
# This exploratory analysis combines data quality assessment, customer behavior, financial exposure, statistical testing and business segmentation to investigate customer churn.
#
# ### Data Understanding & Quality
#
# The integrated dataset contains 7,043 unique customers and 41 variables, with no duplicate customer IDs. Missing values in Offer and Internet Type match the original "None" category
# counts, suggesting an import-related issue.
#
# ### Target Variable Analysis
#
# After excluding newly joined customers, the eligible population contains 6,589 customers, with an observed churn rate of 28.37%.
# Class imbalance highlights the limitations of accuracy alone as a model evaluation metric.
#
# ### Customer Demographics & Geography
#
# Gender shows limited association with churn, while customers aged 65+ and those without dependents show higher observed churn rates.
# San Diego stands out with a churn rate of 66.5%, accounting for 9.9% of all observed churn cases.
#
# ### Customer Behaviour & Services
#
# Customers with 1-6 months of tenure show a churn rate of 77.2%, while monthly contracts account for 88.6% of all churn cases.
# Technical support and online security are associated with substantially lower churn among internet customers.
#
# ### Financial Analysis
#
# Churned customers have 18.2% higher average monthly charges, representing 31.9% of total observed monthly charges.
# Monthly contracts and fiber optic customers concentrate 86.9% and 78.2% of observed financial exposure, respectively.
# Total observed monthly charge exposure amounts to $139,130.85.
#
# ### Statistical & Multivariate Analysis
#
# Contract and tenure show the strongest categorical associations with churn, with Cramer's V of 0.507 and 0.500, respectively.
# Stratified analysis reveals that offer and pricing patterns depend on customer tenure and internet technology, highlighting potential confounding factors.
#
# ### Customer Segmentation & Retention Insights
#
# Monthly customers with early tenure show the highest observed churn rate among selected segments (81.3%), while internet customers without
# premium technical support account for 79.6% of observed financial exposure.
# Competition accounts for 45.0% of reported churn cases, suggesting that retention strategies should consider competitive positioning and service quality alongside pricing.
#
# ### Overall Conclusion
#
# Churn is associated with a combination of customer lifecycle, contractual structure, service adoption and competitive positioning.
# Combining churn rate, churn share, association strength and financial exposure provides an evidence-based foundation for predictive modeling and customer retention prioritization.
#
# ### Limitations
#
# The analysis is exploratory and based on a single-period snapshot, meaning that observed associations do not establish causality.
# Financial exposure is historical and customer segments may overlap.
# Modeling-related findings require reassessment on training data and verification of feature availability before deployment.
