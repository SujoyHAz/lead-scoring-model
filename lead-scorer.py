import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# 1. LOAD DATA

df = pd.read_csv("data/Leads.csv")

columns_to_keep = [
    "Prospect ID",
    "Lead Origin",
    "Lead Source",
    "Converted",
    "TotalVisits",
    "Total Time Spent on Website",
    "Page Views Per Visit",
    "Last Activity",
    "Lead Quality"
]

df = df[columns_to_keep]


# 2. CLEAN DATA

df["TotalVisits"] = df["TotalVisits"].fillna(0)
df["Page Views Per Visit"] = df["Page Views Per Visit"].fillna(0)
df["Lead Source"] = df["Lead Source"].fillna("Unknown")
df["Last Activity"] = df["Last Activity"].fillna("Unknown")
df["Lead Quality"] = df["Lead Quality"].fillna("Unknown")
df["Lead Origin"] = df["Lead Origin"].fillna("Unknown")


# 3. PREPARE FEATURES FOR ML

le = LabelEncoder()

df["Lead Origin encoded"]    = le.fit_transform(df["Lead Origin"])
df["Lead Source encoded"]    = le.fit_transform(df["Lead Source"])
df["Last Activity encoded"]  = le.fit_transform(df["Last Activity"])
df["Lead Quality encoded"]   = le.fit_transform(df["Lead Quality"])


features = [
    "TotalVisits",
    "Total Time Spent on Website",
    "Page Views Per Visit",
    "Lead Origin encoded",
    "Lead Source encoded",
    "Last Activity encoded",
    "Lead Quality encoded"
]

X = df[features]       # inputs (what the model learns from)
y = df["Converted"]    # output (what the model is trying to predict)


# 4. SPLIT DATA INTO TRAINING AND TESTING


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training on {len(X_train)} leads")
print(f"Testing on  {len(X_test)} leads")

# 5. TRAIN THE MODEL

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nModel trained.")


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

print(f"\nModel accuracy: {accuracy:.1f}%")
print("\nDetailed report:")
print(classification_report(y_test, y_pred, target_names=["Not Converted", "Converted"]))



df["conversion_probability"] = model.predict_proba(X)[:, 1]
df["total_score"] = (df["conversion_probability"] * 100).round(1)

def priority_label(score):
    if score >= 70:
        return "Hot"
    elif score >= 40:
        return "Warm"
    else:
        return "Cold"

df["priority"] = df["total_score"].apply(priority_label)

df_sorted = df.sort_values("total_score", ascending=False).reset_index(drop=True)
df_sorted["rank"] = df_sorted.index + 1


os.makedirs("output", exist_ok=True)
df_sorted.to_csv("output/scored_leads.csv", index=False)

print("\n=== LEAD SCORING COMPLETE ===")
print(df_sorted[["rank", "Prospect ID", "total_score", "priority"]].head(10).to_string(index=False))
print(f"\nTotal leads scored: {len(df_sorted)}")
print(f"Hot leads:  {len(df_sorted[df_sorted['priority'] == 'Hot'])}")
print(f"Warm leads: {len(df_sorted[df_sorted['priority'] == 'Warm'])}")
print(f"Cold leads: {len(df_sorted[df_sorted['priority'] == 'Cold'])}")


sns.set_theme(style="whitegrid")
colors = ["#e63946", "#f4a261", "#457b9d"]

# Chart 1: Priority distribution
plt.figure(figsize=(8, 5))
priority_counts = df_sorted["priority"].value_counts().reindex(["Hot", "Warm", "Cold"])
sns.barplot(x=priority_counts.index, y=priority_counts.values, hue=priority_counts.index, palette=colors, legend=False)
plt.title("Lead Priority Distribution", fontsize=14, fontweight="bold")
plt.xlabel("Priority")
plt.ylabel("Number of Leads")
for i, v in enumerate(priority_counts.values):
    plt.text(i, v + 20, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("output/chart_priority_distribution.png", dpi=150)
plt.close()

# Chart 2: Conversion rate by priority
conv_rate = df_sorted.groupby("priority")["Converted"].mean() * 100
conv_rate = conv_rate.reindex(["Hot", "Warm", "Cold"])
plt.figure(figsize=(8, 5))
sns.barplot(x=conv_rate.index, y=conv_rate.values, hue=conv_rate.index, palette=colors, legend=False)
plt.title("Conversion Rate by Priority Label", fontsize=14, fontweight="bold")
plt.xlabel("Priority")
plt.ylabel("Conversion Rate (%)")
for i, v in enumerate(conv_rate.values):
    plt.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("output/chart_conversion_rate.png", dpi=150)
plt.close()

# Chart 3: Score distribution
plt.figure(figsize=(10, 5))
sns.histplot(df_sorted["total_score"], bins=30, color="#457b9d", edgecolor="white")
plt.title("Distribution of Lead Scores", fontsize=14, fontweight="bold")
plt.xlabel("Score (0-100)")
plt.ylabel("Number of Leads")
plt.tight_layout()
plt.savefig("output/chart_score_distribution.png", dpi=150)
plt.close()

print("\nAll charts saved to output/ folder.")