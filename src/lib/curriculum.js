export const curriculum = {
  preWork: {
    id: 'pre-work',
    title: 'Pre-Work: AI Ethics & Responsible AI',
    duration: '~8 hours (self-paced)',
    isNew: true,
    topics: [
      'Introduction to AI ethics and why it matters',
      'Bias in data and algorithms: sources, detection, and mitigation',
      'Fairness metrics and trade-offs in ML systems',
      'Privacy considerations: differential privacy, data anonymization',
      'Transparency and explainability in AI systems',
      'AI governance frameworks and industry standards',
      'Case studies: real-world AI ethics failures and lessons learned'
    ],
    asyncWork: [
      'Readings on responsible AI frameworks',
      'Reflection exercise: analyze an AI system for potential ethical issues',
      'Pre-assessment quiz'
    ],
    resources: [
      { title: 'Google AI Principles', url: 'https://ai.google/responsibility/principles/', type: 'article' },
      { title: 'Anthropic Core Views on AI Safety', url: 'https://www.anthropic.com/news/core-views-on-ai-safety', type: 'article' },
      { title: 'AI Ethics Course - Coursera', url: 'https://www.coursera.org/learn/ai-ethics', type: 'course' },
      { title: 'Fairness in ML - Google', url: 'https://developers.google.com/machine-learning/fairness-overview', type: 'tutorial' }
    ],
    exercise: {
      id: 'ex-prework',
      title: 'AI Ethics Case Analysis',
      description: 'Analyze a hiring dataset for potential biases. This mirrors real-world AI ethics challenges like the Amazon hiring algorithm case.',
      starterCode: `import pandas as pd
import numpy as np

# Sample hiring dataset - similar to real AI ethics cases
data = {
    'age': [25, 35, 45, 28, 52, 33, 41, 29, 38, 55],
    'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
    'years_exp': [2, 8, 15, 3, 20, 7, 12, 4, 10, 22],
    'hired': [1, 1, 0, 1, 0, 1, 0, 1, 1, 0]
}
df = pd.DataFrame(data)

# TODO: Analyze this dataset for potential biases
# 1. Check hiring rates by gender
# 2. Check hiring rates by age groups
# 3. Identify any concerning patterns
# 4. Recommend mitigation strategies

# Your code here:
print("Hiring rate by gender:")
print(df.groupby('gender')['hired'].mean())

# Add more analysis...
`,
      solution: `import pandas as pd
import numpy as np

data = {
    'age': [25, 35, 45, 28, 52, 33, 41, 29, 38, 55],
    'gender': ['M', 'F', 'F', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
    'years_exp': [2, 8, 15, 3, 20, 7, 12, 4, 10, 22],
    'hired': [1, 1, 0, 1, 0, 1, 0, 1, 1, 0]
}
df = pd.DataFrame(data)

# 1. Hiring rates by gender
print("Hiring rate by gender:")
print(df.groupby('gender')['hired'].mean())

# 2. Hiring rates by age group
df['age_group'] = pd.cut(df['age'], bins=[20, 30, 40, 50, 60], labels=['20-30', '30-40', '40-50', '50-60'])
print("\\nHiring rate by age group:")
print(df.groupby('age_group')['hired'].mean())

# 3. Correlation analysis
print("\\nCorrelation with hiring decision:")
print(f"Age: {df['age'].corr(df['hired']):.3f}")
print(f"Years exp: {df['years_exp'].corr(df['hired']):.3f}")

# 4. Findings & Recommendations
print("\\n--- FINDINGS ---")
print("- Younger candidates (20-30) have higher hiring rates")
print("- This could indicate age bias in the hiring process")
print("\\n--- RECOMMENDATIONS ---")
print("1. Blind resume screening (remove age indicators)")
print("2. Structured interviews with standardized scoring")
print("3. Regular bias audits on hiring outcomes")`
    }
  },
  modules: [
    {
      id: 'module-1',
      title: 'Module 1: Foundations (Weeks 1-3)',
      weeks: [
        {
          id: 'week-1',
          title: 'Week 1: What Is AI?',
          sessions: [
            'History and landscape of AI, ML, deep learning, generative AI—how they relate',
            'How machines "learn" from data (intuition-focused, minimal math)'
          ],
          asyncWork: [
            'Readings on AI applications across industries',
            'Reflection exercise on AI in your own field',
            'Connect learnings to pre-work ethics module'
          ],
          resources: [
            { title: 'Stanford CS229 - ML Course', url: 'https://cs229.stanford.edu/', type: 'course' },
            { title: 'Elements of AI - Free Course', url: 'https://www.elementsofai.com/', type: 'course' },
            { title: '3Blue1Brown - Neural Networks', url: 'https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi', type: 'video' }
          ],
          exercise: null
        },
        {
          id: 'week-2',
          title: 'Week 2: Data Thinking',
          sessions: [
            'What is data? Types, sources, quality issues, biases',
            'Exploratory data analysis concepts, summary statistics, visualization basics'
          ],
          asyncWork: [
            'Analyze the IBM HR Attrition dataset',
            'Identify patterns and create visualizations',
            'Document data quality issues and business recommendations'
          ],
          resources: [
            { title: 'Pandas Documentation', url: 'https://pandas.pydata.org/docs/', type: 'docs' },
            { title: 'Kaggle - Data Cleaning', url: 'https://www.kaggle.com/learn/data-cleaning', type: 'tutorial' },
            { title: 'Matplotlib Tutorial', url: 'https://matplotlib.org/stable/tutorials/index.html', type: 'tutorial' },
            { title: 'IBM HR Attrition Dataset', url: 'https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset', type: 'tool' }
          ],
          exercise: {
            id: 'ex-week2',
            title: 'EDA: IBM HR Attrition Analysis',
            description: 'Analyze the IBM HR dataset to identify factors contributing to employee attrition. Business goal: Reduce turnover costs (~$15K per employee lost).',
            starterCode: `import pandas as pd
import numpy as np

# IBM HR Attrition Dataset (simplified)
# Full dataset: kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
np.random.seed(42)
n = 500

data = {
    'Age': np.random.randint(18, 60, n),
    'MonthlyIncome': np.random.randint(1000, 20000, n),
    'YearsAtCompany': np.random.randint(0, 40, n),
    'JobSatisfaction': np.random.randint(1, 5, n),  # 1=Low, 4=Very High
    'WorkLifeBalance': np.random.randint(1, 5, n),
    'OverTime': np.random.choice(['Yes', 'No'], n),
    'Department': np.random.choice(['Sales', 'R&D', 'HR'], n),
    'Attrition': np.random.choice(['Yes', 'No'], n, p=[0.16, 0.84])
}
df = pd.DataFrame(data)

# TODO: Perform EDA to answer business questions:
# 1. What is the overall attrition rate?
# 2. How does attrition vary by department and overtime status?
# 3. What's the relationship between job satisfaction and attrition?
# 4. Provide 3 actionable business recommendations

print("Dataset Shape:", df.shape)
print("\\nAttrition Distribution:")
print(df['Attrition'].value_counts(normalize=True))

# Your analysis here...
`,
            solution: `import pandas as pd
import numpy as np

np.random.seed(42)
n = 500
data = {
    'Age': np.random.randint(18, 60, n),
    'MonthlyIncome': np.random.randint(1000, 20000, n),
    'YearsAtCompany': np.random.randint(0, 40, n),
    'JobSatisfaction': np.random.randint(1, 5, n),
    'WorkLifeBalance': np.random.randint(1, 5, n),
    'OverTime': np.random.choice(['Yes', 'No'], n),
    'Department': np.random.choice(['Sales', 'R&D', 'HR'], n),
    'Attrition': np.random.choice(['Yes', 'No'], n, p=[0.16, 0.84])
}
df = pd.DataFrame(data)

# 1. Overall attrition rate
attrition_rate = (df['Attrition'] == 'Yes').mean() * 100
print(f"Overall Attrition Rate: {attrition_rate:.1f}%")

# 2. Attrition by department and overtime
print("\\nAttrition by Department:")
print(df.groupby('Department')['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100).round(1))

print("\\nAttrition by Overtime Status:")
print(df.groupby('OverTime')['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100).round(1))

# 3. Attrition by job satisfaction
print("\\nAttrition by Job Satisfaction (1=Low, 4=High):")
print(df.groupby('JobSatisfaction')['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100).round(1))

# 4. Business Recommendations
employees_lost = int(attrition_rate/100 * n)
cost_per_employee = 15000
print("\\n" + "="*50)
print("BUSINESS RECOMMENDATIONS")
print("="*50)
print(f"\\nCurrent Impact: {employees_lost} employees lost × ${cost_per_employee:,} = ${employees_lost * cost_per_employee:,}/year")
print("\\n1. OVERTIME POLICY: Review mandatory overtime - correlates with higher attrition")
print("2. SATISFACTION FOCUS: Implement stay interviews for low-satisfaction employees")  
print("3. EARLY INTERVENTION: Target employees in first 2 years (highest risk)")
print(f"\\nPotential savings if 50% reduction: ${int(employees_lost * 0.5 * cost_per_employee):,}/year")`
          }
        },
        {
          id: 'week-3',
          title: 'Week 3: Python, SQL & Tools Setup',
          sessions: [
            'Python basics—variables, data types, lists, loops, functions (Google Colab)',
            'SQL fundamentals—SELECT, WHERE, JOIN, GROUP BY, aggregations; Python-database connectivity'
          ],
          asyncWork: [
            'Complete Python and SQL exercises',
            'Set up working environment (Colab, Git)',
            'Practice data extraction from SQLite database',
            'TechMart project: $2.4M transaction analysis'
          ],
          resources: [
            { title: 'Python Official Tutorial', url: 'https://docs.python.org/3/tutorial/', type: 'docs' },
            { title: 'SQLBolt - Interactive SQL', url: 'https://sqlbolt.com/', type: 'tutorial' },
            { title: 'Mode SQL Tutorial', url: 'https://mode.com/sql-tutorial/', type: 'tutorial' },
            { title: 'Google Colab', url: 'https://colab.research.google.com/', type: 'tool' }
          ],
          exercise: {
            id: 'ex-week3',
            title: 'TechMart SQL Data Pipeline',
            description: 'Analyze $2.4M in e-commerce transactions. Practice SQL queries, CTEs, window functions, and customer segmentation.',
            starterCode: `# TechMart E-Commerce Analysis
# Simulating SQL operations with pandas (real project uses SQLite)

import pandas as pd
import numpy as np

np.random.seed(42)

# Simulated transaction data
n_transactions = 1000
data = {
    'transaction_id': range(1, n_transactions + 1),
    'customer_id': np.random.randint(1, 201, n_transactions),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], n_transactions),
    'amount': np.random.uniform(10, 500, n_transactions).round(2),
    'transaction_date': pd.date_range('2025-01-01', periods=n_transactions, freq='H')
}
df = pd.DataFrame(data)

print(f"Total Revenue: ${df['amount'].sum():,.2f}")

# TODO: Write queries (using pandas) to answer:
# 1. Revenue by product category
# 2. Top 10 customers by total spend
# 3. Monthly revenue trend
# 4. Customer segmentation (High/Medium/Low value)

# Your code here:
`,
            solution: `import pandas as pd
import numpy as np

np.random.seed(42)
n_transactions = 1000
data = {
    'transaction_id': range(1, n_transactions + 1),
    'customer_id': np.random.randint(1, 201, n_transactions),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], n_transactions),
    'amount': np.random.uniform(10, 500, n_transactions).round(2),
    'transaction_date': pd.date_range('2025-01-01', periods=n_transactions, freq='H')
}
df = pd.DataFrame(data)

print(f"Total Revenue: ${df['amount'].sum():,.2f}")
print(f"Total Transactions: {len(df):,}")

# 1. Revenue by category (SQL: GROUP BY product_category)
print("\\n1. REVENUE BY CATEGORY:")
print(df.groupby('product_category')['amount'].agg(['sum', 'mean', 'count']).round(2))

# 2. Top 10 customers (SQL: ORDER BY total DESC LIMIT 10)
print("\\n2. TOP 10 CUSTOMERS:")
top_customers = df.groupby('customer_id')['amount'].sum().nlargest(10)
print(top_customers.round(2))

# 3. Monthly trend (SQL: DATE_TRUNC + GROUP BY)
df['month'] = df['transaction_date'].dt.to_period('M')
print("\\n3. MONTHLY REVENUE:")
print(df.groupby('month')['amount'].sum().round(2))

# 4. Customer segmentation (SQL: CASE WHEN with window function)
customer_totals = df.groupby('customer_id')['amount'].sum()
def segment(total):
    if total > 1000: return 'High Value'
    elif total > 500: return 'Medium Value'
    else: return 'Low Value'

segments = customer_totals.apply(segment).value_counts()
print("\\n4. CUSTOMER SEGMENTS:")
print(segments)`
          }
        }
      ]
    },
    {
      id: 'module-2',
      title: 'Module 2: Core Machine Learning (Weeks 4-7)',
      weeks: [
        {
          id: 'week-4',
          title: 'Week 4: Supervised Learning I — Regression',
          sessions: [
            'The prediction problem, features vs. targets, regression intuition',
            'Linear regression hands-on, interpreting coefficients, sklearn implementation'
          ],
          asyncWork: [
            'Build regression models on Lending Club data',
            'Experiment with feature engineering',
            'Interpret model coefficients for business insights'
          ],
          resources: [
            { title: 'Scikit-learn Linear Models', url: 'https://scikit-learn.org/stable/modules/linear_model.html', type: 'docs' },
            { title: 'StatQuest - Linear Regression', url: 'https://www.youtube.com/watch?v=nk2CQITm_eo', type: 'video' },
            { title: 'Kaggle - Intro to ML', url: 'https://www.kaggle.com/learn/intro-to-machine-learning', type: 'course' },
            { title: 'Lending Club Dataset', url: 'https://www.kaggle.com/datasets/wordsforthewise/lending-club', type: 'tool' }
          ],
          exercise: {
            id: 'ex-week4',
            title: 'Lending Club: Interest Rate Prediction',
            description: 'Predict interest rates on 890K+ loans using regression. Business impact: Better risk-based pricing for lenders.',
            starterCode: `import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Lending Club loan data (simplified)
# Full dataset: 890K+ loans at kaggle.com/datasets/wordsforthewise/lending-club
np.random.seed(42)
n = 1000

# Features that influence interest rate
loan_amount = np.random.randint(1000, 40000, n)
annual_income = np.random.randint(30000, 150000, n)
debt_to_income = np.random.uniform(0, 40, n)
credit_score = np.random.randint(600, 850, n)

# Interest rate (target) - influenced by features
interest_rate = 5 + (40000 - loan_amount) * 0.0001 + \\
                (850 - credit_score) * 0.02 + \\
                debt_to_income * 0.1 + \\
                np.random.randn(n) * 2

X = np.column_stack([loan_amount, annual_income, debt_to_income, credit_score])
y = interest_rate

# TODO: Build regression model
# 1. Split data (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train linear regression
model = LinearRegression()
# Your code here

# 3. Evaluate with RMSE and R²
# Your code here

# 4. Interpret coefficients - which features matter most?
# Your code here
`,
            solution: `import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
n = 1000

loan_amount = np.random.randint(1000, 40000, n)
annual_income = np.random.randint(30000, 150000, n)
debt_to_income = np.random.uniform(0, 40, n)
credit_score = np.random.randint(600, 850, n)

interest_rate = 5 + (40000 - loan_amount) * 0.0001 + \\
                (850 - credit_score) * 0.02 + \\
                debt_to_income * 0.1 + \\
                np.random.randn(n) * 2

X = np.column_stack([loan_amount, annual_income, debt_to_income, credit_score])
y = interest_rate
feature_names = ['loan_amount', 'annual_income', 'debt_to_income', 'credit_score']

# 1. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 3. Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}% interest rate")
print(f"R² Score: {r2:.4f}")

# 4. Interpret coefficients
print("\\nFeature Coefficients:")
for name, coef in zip(feature_names, model.coef_):
    print(f"  {name}: {coef:.6f}")

print("\\nInterpretation:")
print("- Higher credit score → Lower interest rate (negative coefficient)")
print("- Higher debt-to-income → Higher interest rate (positive coefficient)")`
          }
        },
        {
          id: 'week-5',
          title: 'Week 5: Supervised Learning II — Classification',
          sessions: [
            'Classification problems, logistic regression, decision boundaries',
            'Decision trees, random forests (intuition and application)'
          ],
          asyncWork: [
            'Classification exercises with Lending Club default prediction',
            'Compare model approaches (logistic regression vs tree-based)',
            'Handle class imbalance techniques'
          ],
          resources: [
            { title: 'Scikit-learn Classification', url: 'https://scikit-learn.org/stable/modules/tree.html', type: 'docs' },
            { title: 'StatQuest - Decision Trees', url: 'https://www.youtube.com/watch?v=_L39rN6gz7Y', type: 'video' },
            { title: 'StatQuest - Random Forests', url: 'https://www.youtube.com/watch?v=J4Wdy0Wc_xQ', type: 'video' }
          ],
          exercise: {
            id: 'ex-week5',
            title: 'Lending Club: Default Prediction',
            description: 'Build a classifier to predict loan defaults (AUC target: 0.75+). Business impact: Each prevented default saves ~$9,000.',
            starterCode: `import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Lending Club default prediction
# Business context: Average loss per default = $9,000
np.random.seed(42)
n = 2000

# Features
credit_score = np.random.randint(550, 850, n)
debt_to_income = np.random.uniform(0, 50, n)
loan_amount = np.random.randint(1000, 40000, n)
employment_years = np.random.randint(0, 30, n)

# Default probability (imbalanced: ~15% default rate)
default_prob = 1 / (1 + np.exp(-(
    -5 + 
    (700 - credit_score) * 0.02 + 
    debt_to_income * 0.05 - 
    employment_years * 0.1
)))
default = (np.random.random(n) < default_prob).astype(int)

X = np.column_stack([credit_score, debt_to_income, loan_amount, employment_years])
y = default

print(f"Default rate: {y.mean()*100:.1f}%")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TODO: Build and compare classifiers
# 1. Logistic Regression
# 2. Decision Tree
# 3. Random Forest
# 4. Compare using AUC score (target: 0.75+)
`,
            solution: `import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

np.random.seed(42)
n = 2000

credit_score = np.random.randint(550, 850, n)
debt_to_income = np.random.uniform(0, 50, n)
loan_amount = np.random.randint(1000, 40000, n)
employment_years = np.random.randint(0, 30, n)

default_prob = 1 / (1 + np.exp(-(-5 + (700 - credit_score) * 0.02 + debt_to_income * 0.05 - employment_years * 0.1)))
default = (np.random.random(n) < default_prob).astype(int)

X = np.column_stack([credit_score, debt_to_income, loan_amount, employment_years])
y = default

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Default rate: {y.mean()*100:.1f}%")
print("\\n" + "="*50)

# 1. Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
log_pred = log_reg.predict_proba(X_test)[:, 1]
print(f"Logistic Regression AUC: {roc_auc_score(y_test, log_pred):.3f}")

# 2. Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict_proba(X_test)[:, 1]
print(f"Decision Tree AUC: {roc_auc_score(y_test, dt_pred):.3f}")

# 3. Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict_proba(X_test)[:, 1]
print(f"Random Forest AUC: {roc_auc_score(y_test, rf_pred):.3f}")

# Business impact
print("\\n" + "="*50)
print("BUSINESS IMPACT")
defaults_in_test = y_test.sum()
cost_per_default = 9000
print(f"Defaults in test set: {defaults_in_test}")
print(f"Potential loss: ${defaults_in_test * cost_per_default:,}")
print(f"If model prevents 50% of defaults: ${int(defaults_in_test * 0.5 * cost_per_default):,} saved")`
          }
        },
        {
          id: 'week-6',
          title: 'Week 6: Model Evaluation & Deployment',
          sessions: [
            'Train/test splits, overfitting, cross-validation, comprehensive metrics',
            'Hyperparameter tuning, model selection, and Streamlit deployment'
          ],
          asyncWork: [
            'Evaluate models from previous weeks with cross-validation',
            'Write up findings and prepare for deployment',
            'Build interactive Streamlit web application'
          ],
          resources: [
            { title: 'Scikit-learn Model Evaluation', url: 'https://scikit-learn.org/stable/modules/model_evaluation.html', type: 'docs' },
            { title: 'StatQuest - Cross Validation', url: 'https://www.youtube.com/watch?v=fSytzGwwBVw', type: 'video' },
            { title: 'Streamlit Documentation', url: 'https://docs.streamlit.io/', type: 'docs' },
            { title: 'Google ML - Metrics', url: 'https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall', type: 'tutorial' }
          ],
          exercise: {
            id: 'ex-week6',
            title: 'Model Evaluation & Streamlit Deployment',
            description: 'Comprehensive model evaluation with cross-validation and deploy as interactive web app using Streamlit.',
            starterCode: `import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

# Generate imbalanced dataset (simulating real-world scenario)
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2,
                          weights=[0.85, 0.15], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# TODO: Complete comprehensive evaluation
# 1. Confusion matrix analysis
# 2. Calculate precision, recall, F1, AUC
# 3. Perform 5-fold cross-validation
# 4. Discuss: When would you prioritize precision vs recall?

print("Class distribution:", np.bincount(y))
`,
            solution: `import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

X, y = make_classification(n_samples=1000, n_features=10, n_classes=2,
                          weights=[0.85, 0.15], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

# 2. Metrics
print(f"\\nAccuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")

# 3. Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(f"\\n5-Fold CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

# 4. Precision vs Recall Trade-off
print("\\n" + "="*50)
print("WHEN TO PRIORITIZE:")
print("- PRECISION: When false positives are costly")
print("  Example: Spam filter (don't want to block real emails)")
print("- RECALL: When false negatives are costly")
print("  Example: Disease screening (don't want to miss sick patients)")`
          }
        },
        {
          id: 'week-7',
          title: 'Week 7: Unsupervised Learning & Recommendations',
          isRevised: true,
          sessions: [
            'Clustering with K-Means, dimensionality reduction with PCA, similarity measures',
            'Recommendation systems—collaborative filtering, content-based, matrix factorization (SVD)'
          ],
          asyncWork: [
            'Build MovieLens recommendation system',
            'Practice preprocessing pipeline',
            'Handle cold start problem and evaluate with RMSE'
          ],
          resources: [
            { title: 'Scikit-learn Clustering', url: 'https://scikit-learn.org/stable/modules/clustering.html', type: 'docs' },
            { title: 'Google - Recommendation Systems', url: 'https://developers.google.com/machine-learning/recommendation', type: 'course' },
            { title: 'Surprise Library', url: 'https://surpriselib.com/', type: 'docs' },
            { title: 'MovieLens Dataset', url: 'https://grouplens.org/datasets/movielens/', type: 'tool' }
          ],
          exercise: {
            id: 'ex-week7',
            title: 'MovieLens Recommendation System',
            description: 'Build a Netflix-style recommender using MovieLens 100K ratings. Implement collaborative filtering and evaluate with RMSE.',
            starterCode: `import numpy as np

# MovieLens-style ratings matrix
# Full dataset: grouplens.org/datasets/movielens/
# Rows = Users, Columns = Movies, Values = Ratings (0 = not rated)

np.random.seed(42)
n_users, n_movies = 50, 20

# Simulate sparse ratings matrix (most entries are 0)
ratings = np.zeros((n_users, n_movies))
for i in range(n_users):
    # Each user rates 5-10 random movies
    rated_movies = np.random.choice(n_movies, np.random.randint(5, 11), replace=False)
    ratings[i, rated_movies] = np.random.randint(1, 6, len(rated_movies))

movies = [f'Movie_{i}' for i in range(n_movies)]

print(f"Ratings matrix shape: {ratings.shape}")
print(f"Sparsity: {(ratings == 0).mean()*100:.1f}% empty")
print(f"\\nSample ratings (User 0):")
print([(movies[i], int(ratings[0, i])) for i in range(5) if ratings[0, i] > 0])

# TODO: Build collaborative filtering recommender
# 1. Calculate user-user similarity (cosine or correlation)
# 2. Predict rating for a user-movie pair
# 3. Generate top-5 recommendations for a user
# 4. Discuss the cold-start problem
`,
            solution: `import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(42)
n_users, n_movies = 50, 20

ratings = np.zeros((n_users, n_movies))
for i in range(n_users):
    rated_movies = np.random.choice(n_movies, np.random.randint(5, 11), replace=False)
    ratings[i, rated_movies] = np.random.randint(1, 6, len(rated_movies))

movies = [f'Movie_{i}' for i in range(n_movies)]

print(f"Sparsity: {(ratings == 0).mean()*100:.1f}% empty")

# 1. User-user similarity
# Replace 0s with NaN for correlation, then back to 0
ratings_for_sim = np.where(ratings == 0, np.nan, ratings)
user_means = np.nanmean(ratings_for_sim, axis=1, keepdims=True)
ratings_centered = np.nan_to_num(ratings_for_sim - user_means)
user_sim = cosine_similarity(ratings_centered)
np.fill_diagonal(user_sim, 0)  # Don't compare user to self

print(f"\\nUser similarity matrix shape: {user_sim.shape}")

# 2. Predict rating for User 0, Movie 5 (if not rated)
target_user, target_movie = 0, 5
if ratings[target_user, target_movie] == 0:
    # Find users who rated this movie
    rated_by = ratings[:, target_movie] > 0
    if rated_by.sum() > 0:
        sims = user_sim[target_user, rated_by]
        movie_ratings = ratings[rated_by, target_movie]
        if sims.sum() != 0:
            pred = np.dot(sims, movie_ratings) / np.abs(sims).sum()
            print(f"\\nPredicted rating for User {target_user}, {movies[target_movie]}: {pred:.2f}")

# 3. Top-5 recommendations for User 0
user_ratings = ratings[target_user]
unrated = np.where(user_ratings == 0)[0]
predictions = []
for movie_idx in unrated:
    rated_by = ratings[:, movie_idx] > 0
    if rated_by.sum() > 0:
        sims = user_sim[target_user, rated_by]
        if np.abs(sims).sum() > 0:
            pred = np.dot(sims, ratings[rated_by, movie_idx]) / np.abs(sims).sum()
            predictions.append((movies[movie_idx], pred))

predictions.sort(key=lambda x: x[1], reverse=True)
print(f"\\nTop 5 Recommendations for User {target_user}:")
for movie, score in predictions[:5]:
    print(f"  {movie}: {score:.2f}")

# 4. Cold Start Problem
print("\\n" + "="*50)
print("COLD START PROBLEM:")
print("- New users: No ratings → can't find similar users")
print("- New movies: No ratings → can't recommend")
print("\\nSOLUTIONS:")
print("- Content-based filtering for new items")
print("- Ask new users for preferences (onboarding)")
print("- Hybrid approaches")`
          }
        }
      ]
    },
    {
      id: 'module-3',
      title: 'Module 3: Deep Learning & Neural Networks (Weeks 8-9)',
      weeks: [
        {
          id: 'week-8',
          title: 'Week 8: Neural Network Fundamentals',
          hasCapstone: true,
          sessions: [
            'What is a neural network? Layers, weights, activation functions (visual intuition)',
            'Training networks—loss functions, gradient descent, backpropagation (conceptual), Keras implementation'
          ],
          asyncWork: [
            'Experiment with neural networks in Keras/TensorFlow',
            'Tune hyperparameters (layers, neurons, learning rate)',
            '★ Capstone Project: Receive guidelines and begin planning'
          ],
          resources: [
            { title: 'TensorFlow Tutorials', url: 'https://www.tensorflow.org/tutorials', type: 'tutorial' },
            { title: 'Keras Documentation', url: 'https://keras.io/guides/', type: 'docs' },
            { title: '3Blue1Brown - Neural Networks', url: 'https://www.youtube.com/watch?v=aircAruvnKk', type: 'video' },
            { title: 'UCI Heart Disease Dataset', url: 'https://archive.ics.uci.edu/dataset/45/heart+disease', type: 'tool' }
          ],
          exercise: {
            id: 'ex-week8',
            title: 'Heart Disease Prediction with Neural Networks',
            description: 'Build a neural network to predict heart disease using UCI dataset. Business impact: ~$75K per missed diagnosis. Compare NN vs traditional ML.',
            starterCode: `import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# UCI Heart Disease Dataset (simplified)
# Full dataset: archive.ics.uci.edu/dataset/45/heart+disease
np.random.seed(42)
n = 500

# Features (simplified from original 13 features)
age = np.random.randint(30, 80, n)
cholesterol = np.random.randint(150, 400, n)
max_heart_rate = np.random.randint(80, 200, n)
blood_pressure = np.random.randint(90, 180, n)
chest_pain_type = np.random.randint(0, 4, n)

# Target: heart disease (influenced by features)
disease_prob = 1 / (1 + np.exp(-(
    -3 + age * 0.05 + cholesterol * 0.005 - max_heart_rate * 0.02 + blood_pressure * 0.01
)))
heart_disease = (np.random.random(n) < disease_prob).astype(int)

X = np.column_stack([age, cholesterol, max_heart_rate, blood_pressure, chest_pain_type])
y = heart_disease

print(f"Disease prevalence: {y.mean()*100:.1f}%")
print(f"Business context: ~$75K cost per missed diagnosis")

# TODO: 
# 1. Scale features
# 2. Compare Logistic Regression baseline vs Neural Network
# 3. Build a simple NN (conceptually - describe architecture)
# 4. Discuss: When would you choose NN over logistic regression?
`,
            solution: `import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

np.random.seed(42)
n = 500

age = np.random.randint(30, 80, n)
cholesterol = np.random.randint(150, 400, n)
max_heart_rate = np.random.randint(80, 200, n)
blood_pressure = np.random.randint(90, 180, n)
chest_pain_type = np.random.randint(0, 4, n)

disease_prob = 1 / (1 + np.exp(-(-3 + age * 0.05 + cholesterol * 0.005 - max_heart_rate * 0.02 + blood_pressure * 0.01)))
heart_disease = (np.random.random(n) < disease_prob).astype(int)

X = np.column_stack([age, cholesterol, max_heart_rate, blood_pressure, chest_pain_type])
y = heart_disease

# 1. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 2. Baseline: Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
log_pred = log_reg.predict_proba(X_test)[:, 1]
print(f"Logistic Regression AUC: {roc_auc_score(y_test, log_pred):.3f}")

# 3. Neural Network Architecture (conceptual)
print("\\n" + "="*50)
print("NEURAL NETWORK ARCHITECTURE:")
print("="*50)
print("Input Layer: 5 features")
print("Hidden Layer 1: 16 neurons, ReLU activation")
print("Hidden Layer 2: 8 neurons, ReLU activation")
print("Output Layer: 1 neuron, Sigmoid activation")
print("\\nIn Keras:")
print('''
model = Sequential([
    Dense(16, activation='relu', input_shape=(5,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
''')

# 4. When to use NN vs Logistic Regression
print("\\n" + "="*50)
print("WHEN TO CHOOSE:")
print("="*50)
print("LOGISTIC REGRESSION:")
print("  - Small datasets (<1000 samples)")
print("  - Need interpretability (coefficients)")
print("  - Linear relationships")
print("\\nNEURAL NETWORKS:")
print("  - Large datasets (10K+ samples)")
print("  - Complex non-linear patterns")
print("  - Image, text, or sequential data")`
          }
        },
        {
          id: 'week-9',
          title: 'Week 9: Deep Learning Applications',
          sessions: [
            'Convolutional neural networks for images, transfer learning with pre-trained models (VGG16, ResNet, MobileNet)',
            'Sequence models (RNNs, LSTMs), attention mechanisms, introduction to transformers'
          ],
          asyncWork: [
            'Image classification mini-project using transfer learning',
            'Readings on transformer architecture (why it replaced RNNs)',
            'Continue capstone project'
          ],
          resources: [
            { title: 'TensorFlow Transfer Learning', url: 'https://www.tensorflow.org/tutorials/images/transfer_learning', type: 'tutorial' },
            { title: 'Illustrated Transformer', url: 'https://jalammar.github.io/illustrated-transformer/', type: 'article' },
            { title: 'Kaggle Image Classification', url: 'https://www.kaggle.com/competitions?search=image+classification', type: 'tool' }
          ],
          exercise: {
            id: 'ex-week9',
            title: 'Image Classification with Transfer Learning',
            description: 'Use pre-trained models (VGG16/ResNet) for image classification. Demonstrate 80%+ reduction in training time vs training from scratch.',
            starterCode: `# Transfer Learning Concepts
# In practice, use TensorFlow/Keras with pre-trained models

print("="*50)
print("TRANSFER LEARNING WORKFLOW")
print("="*50)

steps = """
1. LOAD PRE-TRAINED MODEL (e.g., VGG16, ResNet50)
   - Trained on ImageNet (14M images, 1000 classes)
   - Remove the top classification layer
   
2. FREEZE BASE LAYERS
   - Keep learned features (edges, textures, shapes)
   - Prevents overwriting useful representations
   
3. ADD CUSTOM CLASSIFICATION HEAD
   - GlobalAveragePooling2D
   - Dense layers for your specific classes
   - Softmax/Sigmoid output
   
4. TRAIN ON YOUR DATA
   - Only train the new layers initially
   - Optionally fine-tune top layers of base
"""
print(steps)

# TODO:
# 1. Explain why freezing layers helps
# 2. When would you fine-tune vs freeze all base layers?
# 3. What's the benefit over training from scratch?
`,
            solution: `print("="*50)
print("TRANSFER LEARNING WORKFLOW")
print("="*50)

print("""
1. LOAD PRE-TRAINED MODEL
   model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
   
2. FREEZE BASE LAYERS
   for layer in model.layers:
       layer.trainable = False
       
3. ADD CUSTOM HEAD
   x = GlobalAveragePooling2D()(model.output)
   x = Dense(256, activation='relu')(x)
   output = Dense(num_classes, activation='softmax')(x)
   
4. COMPILE AND TRAIN
   model.compile(optimizer='adam', loss='categorical_crossentropy')
   model.fit(train_data, epochs=10)
""")

print("\\n" + "="*50)
print("KEY QUESTIONS ANSWERED")
print("="*50)

print("""
1. WHY FREEZE LAYERS?
   - Lower layers learn universal features (edges, colors, textures)
   - These transfer well to any image task
   - Prevents overfitting on small datasets
   - Much faster training (fewer parameters to update)

2. WHEN TO FINE-TUNE VS FREEZE ALL?
   FREEZE ALL:
   - Very small dataset (<1000 images)
   - Target domain similar to ImageNet
   
   FINE-TUNE TOP LAYERS:
   - Larger dataset (1000+ images)
   - Target domain different from ImageNet
   - Want to adapt high-level features

3. BENEFITS OVER TRAINING FROM SCRATCH:
   - 80-90% reduction in training time
   - Works with small datasets (100s vs millions)
   - Often better accuracy (leverages ImageNet knowledge)
   - Less compute required (cost savings)
   
   Example: 
   - From scratch: 100 epochs, 10 hours, 50% accuracy
   - Transfer: 10 epochs, 30 minutes, 90% accuracy
""")`
          }
        }
      ]
    },
    {
      id: 'module-4',
      title: 'Module 4: Generative AI & LLMs (Weeks 10-11)',
      weeks: [
        {
          id: 'week-10',
          title: 'Week 10: LLMs & Prompt Engineering',
          sessions: [
            'How LLMs work—tokenization (BPE), embeddings (semantic similarity), next-token prediction, temperature, scaling laws, LLM landscape (GPT, Claude, LLaMA, Gemini, Mistral)',
            'Capabilities & limitations (hallucinations, knowledge cutoff, reasoning gaps, context limits), prompt engineering (zero-shot, few-shot, chain-of-thought), LLM APIs (OpenAI, Anthropic)'
          ],
          asyncWork: [
            'LLM fundamentals review and prompt engineering practice',
            'API exploration with OpenAI and Anthropic',
            'Build LLM-powered ticket classification system'
          ],
          resources: [
            { title: 'Anthropic - Claude Documentation', url: 'https://docs.anthropic.com/', type: 'docs' },
            { title: 'OpenAI - GPT Best Practices', url: 'https://platform.openai.com/docs/guides/prompt-engineering', type: 'docs' },
            { title: 'Attention Is All You Need (Paper)', url: 'https://arxiv.org/abs/1706.03762', type: 'paper' },
            { title: 'Prompt Engineering Guide', url: 'https://www.promptingguide.ai/', type: 'tutorial' }
          ],
          exercise: {
            id: 'ex-week10',
            title: 'LLM Customer Support System',
            description: 'Build a ticket classification and response system using prompt engineering. Target: 85%+ accuracy on classification, $500K+ annual savings potential.',
            starterCode: `# LLM Customer Support System
# Using prompt engineering for ticket classification and response

# Sample support tickets
tickets = [
    {"id": 1, "text": "I can't log into my account, password reset not working"},
    {"id": 2, "text": "When will my order #12345 arrive? It's been 2 weeks"},
    {"id": 3, "text": "I want a refund for the damaged product I received"},
    {"id": 4, "text": "How do I upgrade my subscription to premium?"},
    {"id": 5, "text": "Your app keeps crashing on my iPhone"}
]

categories = ["Account Issues", "Shipping", "Refunds", "Billing", "Technical Support"]

# TODO: Design prompts for:
# 1. Zero-shot classification
# 2. Few-shot classification (with examples)
# 3. Response generation with constraints

print("="*50)
print("PROMPT ENGINEERING FOR CUSTOMER SUPPORT")
print("="*50)

# Example zero-shot prompt template
zero_shot_prompt = """
Classify this customer support ticket into one category.

Categories: {categories}

Ticket: {ticket_text}

Category:"""

print("Zero-shot prompt template:")
print(zero_shot_prompt)

# Your prompts here...
`,
            solution: `# LLM Customer Support System

tickets = [
    {"id": 1, "text": "I can't log into my account, password reset not working"},
    {"id": 2, "text": "When will my order #12345 arrive? It's been 2 weeks"},
    {"id": 3, "text": "I want a refund for the damaged product I received"},
    {"id": 4, "text": "How do I upgrade my subscription to premium?"},
    {"id": 5, "text": "Your app keeps crashing on my iPhone"}
]

categories = ["Account Issues", "Shipping", "Refunds", "Billing", "Technical Support"]

print("="*60)
print("1. ZERO-SHOT CLASSIFICATION PROMPT")
print("="*60)
print("""
Classify this customer support ticket into exactly one category.

Categories: Account Issues, Shipping, Refunds, Billing, Technical Support

Ticket: "{ticket_text}"

Respond with only the category name, nothing else.
""")

print("="*60)
print("2. FEW-SHOT CLASSIFICATION PROMPT")
print("="*60)
print("""
Classify customer support tickets. Here are examples:

Ticket: "I forgot my password"
Category: Account Issues

Ticket: "Where is my package?"
Category: Shipping

Ticket: "I want my money back"
Category: Refunds

Ticket: "How much does premium cost?"
Category: Billing

Ticket: "The app won't open"
Category: Technical Support

Now classify this ticket:
Ticket: "{ticket_text}"
Category:
""")

print("="*60)
print("3. RESPONSE GENERATION WITH CONSTRAINTS")
print("="*60)
print("""
You are a helpful customer support agent. Generate a response to this ticket.

CONSTRAINTS:
- Be empathetic and professional
- Keep response under 100 words
- Include a specific next step for the customer
- Do not make promises about timelines you can't keep

Ticket Category: {category}
Ticket: "{ticket_text}"

Response:
""")

print("="*60)
print("BUSINESS IMPACT")
print("="*60)
print("""
Current state: 10 agents × $50K salary = $500K/year
With LLM automation (85% auto-resolved):
- Reduced to 4 agents = $200K/year
- Annual savings: $300K+
- Faster response time (seconds vs hours)
- 24/7 availability
""")`
          }
        },
        {
          id: 'week-11',
          title: 'Week 11: RAG & Agentic AI',
          isRevised: true,
          sessions: [
            'Embeddings deep dive—text embeddings at scale, vector similarity search, vector databases (ChromaDB, FAISS, Pinecone), document chunking strategies',
            'RAG architecture—retrieval, augmentation, generation pipeline; LangChain fundamentals; Agentic AI concepts (tool use, function calling, autonomous agents)'
          ],
          asyncWork: [
            'Build a RAG-powered application using LangChain',
            'Implement document Q&A system with vector database',
            'Continue capstone project'
          ],
          resources: [
            { title: 'LangChain Documentation', url: 'https://python.langchain.com/docs/', type: 'docs' },
            { title: 'Pinecone - Vector DB Guide', url: 'https://www.pinecone.io/learn/', type: 'tutorial' },
            { title: 'ChromaDB', url: 'https://docs.trychroma.com/', type: 'docs' },
            { title: 'FAISS by Facebook', url: 'https://faiss.ai/', type: 'docs' }
          ],
          exercise: {
            id: 'ex-week11',
            title: 'Document Q&A System with RAG',
            description: 'Build a RAG-powered knowledge base that answers questions about company documents. Uses ChromaDB/FAISS for vector storage.',
            starterCode: `# RAG (Retrieval-Augmented Generation) Pipeline
# Build a document Q&A system

print("="*60)
print("RAG ARCHITECTURE OVERVIEW")
print("="*60)

# Document corpus (simulated company knowledge base)
documents = [
    {"id": 1, "title": "Return Policy", "content": "Items can be returned within 30 days of purchase. Items must be unused and in original packaging. Refunds are processed within 5-7 business days."},
    {"id": 2, "title": "Shipping Info", "content": "Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days. Free shipping on orders over $50."},
    {"id": 3, "title": "Product Warranty", "content": "All electronics come with a 1-year warranty. Warranty covers manufacturing defects. Accidental damage is not covered."},
    {"id": 4, "title": "Account Security", "content": "Enable two-factor authentication for account security. Passwords must be at least 8 characters. Reset password via email link."},
]

user_question = "How long do I have to return an item?"

# TODO: Implement RAG pipeline steps:
# 1. Document chunking strategy
# 2. Create embeddings (conceptually)
# 3. Store in vector database
# 4. Retrieve relevant chunks
# 5. Augment prompt with context
# 6. Generate answer
`,
            solution: `# RAG (Retrieval-Augmented Generation) Pipeline

documents = [
    {"id": 1, "title": "Return Policy", "content": "Items can be returned within 30 days of purchase. Items must be unused and in original packaging. Refunds are processed within 5-7 business days."},
    {"id": 2, "title": "Shipping Info", "content": "Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days. Free shipping on orders over $50."},
    {"id": 3, "title": "Product Warranty", "content": "All electronics come with a 1-year warranty. Warranty covers manufacturing defects. Accidental damage is not covered."},
    {"id": 4, "title": "Account Security", "content": "Enable two-factor authentication for account security. Passwords must be at least 8 characters. Reset password via email link."},
]

user_question = "How long do I have to return an item?"

print("="*60)
print("RAG PIPELINE IMPLEMENTATION")
print("="*60)

print("""
STEP 1: DOCUMENT CHUNKING
- Split documents into smaller chunks (500-1000 tokens)
- Overlap chunks by 10-20% for context continuity
- Keep metadata (title, source) with each chunk
""")

print("""
STEP 2: CREATE EMBEDDINGS
- Use embedding model (OpenAI ada-002, sentence-transformers)
- Convert text chunks to dense vectors (1536 dimensions)
- Vectors capture semantic meaning
""")

print("""
STEP 3: STORE IN VECTOR DATABASE
- ChromaDB (local, easy setup)
- FAISS (fast, Facebook's library)
- Pinecone (cloud, scalable)
""")

print("""
STEP 4: RETRIEVE RELEVANT CHUNKS
- Convert user question to embedding
- Find top-k similar chunks (cosine similarity)
- Return chunks with highest similarity scores
""")

# Simulate retrieval (in practice, use vector similarity)
retrieved_doc = documents[0]  # Return policy is most relevant
print(f"\\nRetrieved: {retrieved_doc['title']}")
print(f"Content: {retrieved_doc['content']}")

print("""
STEP 5: AUGMENT PROMPT WITH CONTEXT
""")
augmented_prompt = f"""
Answer the user's question based only on the provided context.
If the answer isn't in the context, say "I don't have that information."

Context:
{retrieved_doc['content']}

Question: {user_question}

Answer:"""
print(augmented_prompt)

print("""
STEP 6: GENERATE ANSWER
- Send augmented prompt to LLM
- LLM generates grounded response
- Response based on retrieved context (not hallucinated)

Expected Answer: "You have 30 days to return an item from the date of purchase."
""")

print("\\n" + "="*60)
print("VECTOR DATABASE COMPARISON")
print("="*60)
print("""
| Feature      | ChromaDB | FAISS    | Pinecone |
|--------------|----------|----------|----------|
| Setup        | Easy     | Medium   | Easy     |
| Scalability  | Medium   | High     | High     |
| Cost         | Free     | Free     | Paid     |
| Cloud/Local  | Local    | Local    | Cloud    |
| Best for     | Prototyping | Production | Enterprise |
""")`
          }
        }
      ]
    },
    {
      id: 'module-5',
      title: 'Module 5: Capstone & Professional Application (Week 12)',
      weeks: [
        {
          id: 'week-12',
          title: 'Week 12: Capstone Presentations & Next Steps',
          sessions: [
            'Student capstone presentations (first half)',
            'Student capstone presentations (second half), peer feedback'
          ],
          asyncWork: [
            'Finalize capstone project and GitHub documentation',
            'Prepare 10-minute presentation',
            'Portfolio and resume review',
            'Career paths in AI and continued learning resources'
          ],
          resources: [
            { title: 'GitHub Profile Guide', url: 'https://docs.github.com/en/account-and-profile', type: 'article' },
            { title: 'Data Science Portfolio Tips', url: 'https://www.datacamp.com/blog/how-to-build-a-data-science-portfolio', type: 'article' },
            { title: 'Kaggle Competitions', url: 'https://www.kaggle.com/competitions', type: 'tool' },
            { title: 'LinkedIn for Data Scientists', url: 'https://www.linkedin.com/pulse/how-optimize-your-linkedin-profile-data-science-roles/', type: 'article' }
          ],
          exercise: null
        }
      ]
    }
  ]
};

// Week unlock order - defines which weeks are unlocked by default and progression
export const weekUnlockOrder = [
  'pre-work',  // Always unlocked
  'week-1',    // Always unlocked
  'week-2',    // Unlocked after completing pre-work + week-1
  'week-3',
  'week-4',
  'week-5',
  'week-6',
  'week-7',
  'week-8',
  'week-9',
  'week-10',
  'week-11',
  'week-12'
];

// Quizzes for each week
export const quizzes = {
  'pre-work': {
    id: 'quiz-prework',
    title: 'AI Ethics Quiz',
    passingScore: 70,
    questions: [
      {
        id: 'q1',
        question: 'What is algorithmic bias?',
        options: [
          'A type of computer virus',
          'Systematic errors in AI systems that create unfair outcomes',
          'A programming language',
          'A hardware malfunction'
        ],
        correctAnswer: 1
      },
      {
        id: 'q2',
        question: 'Which of the following is a key principle of responsible AI?',
        options: [
          'Maximizing profit at all costs',
          'Hiding how AI systems make decisions',
          'Transparency and explainability',
          'Using as much data as possible without consent'
        ],
        correctAnswer: 2
      },
      {
        id: 'q3',
        question: 'What is differential privacy?',
        options: [
          'A technique to protect individual data while allowing useful analysis',
          'A way to make AI systems faster',
          'A type of neural network',
          'A programming framework'
        ],
        correctAnswer: 0
      },
      {
        id: 'q4',
        question: 'Why is AI fairness important?',
        options: [
          'It makes code run faster',
          'It ensures AI systems do not discriminate against certain groups',
          'It reduces server costs',
          'It is not important'
        ],
        correctAnswer: 1
      },
      {
        id: 'q5',
        question: 'What should you do if you discover bias in your AI model?',
        options: [
          'Ignore it and deploy anyway',
          'Investigate the source, mitigate it, and document your findings',
          'Delete all the data',
          'Blame the data provider'
        ],
        correctAnswer: 1
      }
    ]
  },
  'week-1': {
    id: 'quiz-week1',
    title: 'What Is AI? Quiz',
    passingScore: 70,
    questions: [
      {
        id: 'q1',
        question: 'What is Machine Learning?',
        options: [
          'A type of robot',
          'A subset of AI where systems learn from data',
          'A programming language',
          'A type of database'
        ],
        correctAnswer: 1
      },
      {
        id: 'q2',
        question: 'Which of these is NOT a type of machine learning?',
        options: [
          'Supervised learning',
          'Unsupervised learning',
          'Reinforcement learning',
          'Mechanical learning'
        ],
        correctAnswer: 3
      },
      {
        id: 'q3',
        question: 'What is Deep Learning?',
        options: [
          'Learning while sleeping',
          'A subset of ML using neural networks with many layers',
          'A type of database query',
          'A hardware component'
        ],
        correctAnswer: 1
      },
      {
        id: 'q4',
        question: 'What does "training" mean in machine learning?',
        options: [
          'Teaching humans to use computers',
          'The process of a model learning patterns from data',
          'Installing software',
          'Writing documentation'
        ],
        correctAnswer: 1
      },
      {
        id: 'q5',
        question: 'Which company created ChatGPT?',
        options: [
          'Google',
          'Meta',
          'OpenAI',
          'Microsoft'
        ],
        correctAnswer: 2
      }
    ]
  },
  'week-2': {
    id: 'quiz-week2',
    title: 'Data Thinking Quiz',
    passingScore: 70,
    questions: [
      {
        id: 'q1',
        question: 'What is Exploratory Data Analysis (EDA)?',
        options: [
          'A programming language',
          'The process of analyzing data sets to summarize their main characteristics',
          'A type of database',
          'A machine learning model'
        ],
        correctAnswer: 1
      },
      {
        id: 'q2',
        question: 'Which is NOT a common data quality issue?',
        options: [
          'Missing values',
          'Duplicate records',
          'Data being too accurate',
          'Inconsistent formatting'
        ],
        correctAnswer: 2
      },
      {
        id: 'q3',
        question: 'What does the median represent?',
        options: [
          'The average value',
          'The most frequent value',
          'The middle value when data is sorted',
          'The range of values'
        ],
        correctAnswer: 2
      },
      {
        id: 'q4',
        question: 'Why is data visualization important?',
        options: [
          'It makes reports longer',
          'It helps identify patterns and communicate insights',
          'It slows down analysis',
          'It is not important'
        ],
        correctAnswer: 1
      },
      {
        id: 'q5',
        question: 'In the IBM HR Attrition project, what is the business goal?',
        options: [
          'Increase employee turnover',
          'Reduce turnover costs (~$15K per employee)',
          'Hire more employees',
          'Eliminate HR department'
        ],
        correctAnswer: 1
      }
    ]
  },
  'week-3': {
    id: 'quiz-week3',
    title: 'Python & SQL Quiz',
    passingScore: 70,
    questions: [
      {
        id: 'q1',
        question: 'What does SQL stand for?',
        options: [
          'Simple Question Language',
          'Structured Query Language',
          'System Quality Level',
          'Standard Query Logic'
        ],
        correctAnswer: 1
      },
      {
        id: 'q2',
        question: 'Which SQL clause is used to filter results?',
        options: [
          'SELECT',
          'FROM',
          'WHERE',
          'ORDER BY'
        ],
        correctAnswer: 2
      },
      {
        id: 'q3',
        question: 'What does a JOIN do in SQL?',
        options: [
          'Deletes tables',
          'Combines rows from two or more tables',
          'Creates a new database',
          'Sorts data'
        ],
        correctAnswer: 1
      },
      {
        id: 'q4',
        question: 'Which Python data structure uses key-value pairs?',
        options: [
          'List',
          'Tuple',
          'Dictionary',
          'Set'
        ],
        correctAnswer: 2
      },
      {
        id: 'q5',
        question: 'In the TechMart project, what was the total transaction value analyzed?',
        options: [
          '$500K',
          '$1.2M',
          '$2.4M',
          '$5M'
        ],
        correctAnswer: 2
      }
    ]
  },
  'week-4': {
    id: 'quiz-week4',
    title: 'Regression Quiz',
    passingScore: 70,
    questions: [
      {
        id: 'q1',
        question: 'What is linear regression used for?',
        options: [
          'Classification problems',
          'Predicting continuous numerical values',
          'Clustering data',
          'Image recognition'
        ],
        correctAnswer: 1
      },
      {
        id: 'q2',
        question: 'What is a "feature" in machine learning?',
        options: [
          'The output we want to predict',
          'An input variable used for prediction',
          'A type of algorithm',
          'A visualization tool'
        ],
        correctAnswer: 1
      },
      {
        id: 'q3',
        question: 'What metric is commonly used to evaluate regression models?',
        options: [
          'Accuracy',
          'RMSE (Root Mean Square Error)',
          'F1 Score',
          'Precision'
        ],
        correctAnswer: 1
      },
      {
        id: 'q4',
        question: 'In the Lending Club project, what are we predicting?',
        options: [
          'Whether a loan will default',
          'Interest rates on loans',
          'Customer satisfaction',
          'Stock prices'
        ],
        correctAnswer: 1
      },
      {
        id: 'q5',
        question: 'What does a negative coefficient mean in linear regression?',
        options: [
          'The model failed',
          'As the feature increases, the target decreases',
          'The feature is not important',
          'An error occurred'
        ],
        correctAnswer: 1
      }
    ]
  },
  'week-5': {
    id: 'quiz-week5',
    title: 'Classification Quiz',
    passingScore: 70,
    questions: [
      {
        id: 'q1',
        question: 'What is classification used for?',
        options: [
          'Predicting continuous values',
          'Predicting categorical outcomes',
          'Grouping similar data',
          'Reducing dimensions'
        ],
        correctAnswer: 1
      },
      {
        id: 'q2',
        question: 'What is a Random Forest?',
        options: [
          'A single decision tree',
          'An ensemble of multiple decision trees',
          'A type of data',
          'A visualization library'
        ],
        correctAnswer: 1
      },
      {
        id: 'q3',
        question: 'What does AUC-ROC measure?',
        options: [
          'Model speed',
          'Classification performance across thresholds',
          'Data size',
          'Feature importance'
        ],
        correctAnswer: 1
      },
      {
        id: 'q4',
        question: 'In loan default prediction, how much does each prevented default save?',
        options: [
          '$1,000',
          '$5,000',
          '$9,000',
          '$50,000'
        ],
        correctAnswer: 2
      },
      {
        id: 'q5',
        question: 'What is class imbalance?',
        options: [
          'When features have different scales',
          'When one class is much more frequent than others',
          'When models are too complex',
          'When data is missing'
        ],
        correctAnswer: 1
      }
    ]
  },
  'week-6': {
    id: 'quiz-week6',
    title: 'Model Evaluation Quiz',
    passingScore: 70,
    questions: [
      {
        id: 'q1',
        question: 'What is overfitting?',
        options: [
          'When a model performs well on new data',
          'When a model learns noise and performs poorly on new data',
          'When a model is too simple',
          'When training takes too long'
        ],
        correctAnswer: 1
      },
      {
        id: 'q2',
        question: 'What is cross-validation used for?',
        options: [
          'Cleaning data',
          'Evaluating model performance more robustly',
          'Visualizing results',
          'Speeding up training'
        ],
        correctAnswer: 1
      },
      {
        id: 'q3',
        question: 'When should you prioritize recall over precision?',
        options: [
          'When false positives are costly',
          'When false negatives are costly (e.g., disease screening)',
          'Always',
          'Never'
        ],
        correctAnswer: 1
      },
      {
        id: 'q4',
        question: 'What is Streamlit used for?',
        options: [
          'Training models',
          'Building interactive web applications',
          'Data cleaning',
          'Database management'
        ],
        correctAnswer: 1
      },
      {
        id: 'q5',
        question: 'What does a confusion matrix show?',
        options: [
          'Model parameters',
          'Prediction results vs actual values (TP, TN, FP, FN)',
          'Training time',
          'Feature correlations'
        ],
        correctAnswer: 1
      }
    ]
  },
  'week-7': {
    id: 'quiz-week7',
    title: 'Unsupervised Learning & Recommendations Quiz',
    passingScore: 70,
    questions: [
      {
        id: 'q1',
        question: 'What is clustering?',
        options: [
          'Predicting labels',
          'Grouping similar data points together',
          'Training neural networks',
          'Data visualization'
        ],
        correctAnswer: 1
      },
      {
        id: 'q2',
        question: 'What is collaborative filtering?',
        options: [
          'Filtering spam emails',
          'Recommending items based on similar users\' preferences',
          'Cleaning data',
          'A clustering method'
        ],
        correctAnswer: 1
      },
      {
        id: 'q3',
        question: 'What is the cold start problem?',
        options: [
          'Servers being too cold',
          'Difficulty recommending for new users/items with no history',
          'Slow computation',
          'Data storage issues'
        ],
        correctAnswer: 1
      },
      {
        id: 'q4',
        question: 'What dataset is used in the Week 7 project?',
        options: [
          'Lending Club',
          'MovieLens 100K',
          'IBM HR',
          'MNIST'
        ],
        correctAnswer: 1
      },
      {
        id: 'q5',
        question: 'What does PCA stand for?',
        options: [
          'Primary Component Analysis',
          'Principal Component Analysis',
          'Partial Cluster Algorithm',
          'Predictive Classification Approach'
        ],
        correctAnswer: 1
      }
    ]
  },
  'week-8': {
    id: 'quiz-week8',
    title: 'Neural Networks Quiz',
    passingScore: 70,
    questions: [
      {
        id: 'q1',
        question: 'What is an activation function?',
        options: [
          'A function that starts the training',
          'A function that introduces non-linearity to the network',
          'A data loading function',
          'A loss calculation'
        ],
        correctAnswer: 1
      },
      {
        id: 'q2',
        question: 'What is backpropagation?',
        options: [
          'Moving data backwards',
          'Algorithm for calculating gradients to update weights',
          'A type of neural network',
          'Data preprocessing'
        ],
        correctAnswer: 1
      },
      {
        id: 'q3',
        question: 'What is the business impact of a missed heart disease diagnosis?',
        options: [
          '$5,000',
          '$25,000',
          '$75,000',
          '$500,000'
        ],
        correctAnswer: 2
      },
      {
        id: 'q4',
        question: 'When should you choose a neural network over logistic regression?',
        options: [
          'Small datasets with linear relationships',
          'Large datasets with complex non-linear patterns',
          'Always',
          'Never'
        ],
        correctAnswer: 1
      },
      {
        id: 'q5',
        question: 'What does ReLU activation do?',
        options: [
          'Returns the input if positive, else returns 0',
          'Squashes values between 0 and 1',
          'Normalizes the data',
          'Calculates the loss'
        ],
        correctAnswer: 0
      }
    ]
  },
  'week-9': {
    id: 'quiz-week9',
    title: 'Deep Learning Applications Quiz',
    passingScore: 70,
    questions: [
      {
        id: 'q1',
        question: 'What is a CNN primarily used for?',
        options: [
          'Text processing',
          'Image and visual data processing',
          'Time series prediction',
          'Recommendation systems'
        ],
        correctAnswer: 1
      },
      {
        id: 'q2',
        question: 'What is transfer learning?',
        options: [
          'Moving data between servers',
          'Using a pre-trained model for a new task',
          'A type of data augmentation',
          'Transferring files'
        ],
        correctAnswer: 1
      },
      {
        id: 'q3',
        question: 'Why freeze base layers in transfer learning?',
        options: [
          'To make training faster and preserve learned features',
          'To increase model size',
          'To add more data',
          'To visualize results'
        ],
        correctAnswer: 0
      },
      {
        id: 'q4',
        question: 'What does the Transformer architecture use instead of RNNs?',
        options: [
          'Convolutional layers',
          'Self-attention mechanisms',
          'Pooling layers',
          'Dropout'
        ],
        correctAnswer: 1
      },
      {
        id: 'q5',
        question: 'How much can transfer learning reduce training time?',
        options: [
          '10-20%',
          '30-50%',
          '80-90%',
          'No reduction'
        ],
        correctAnswer: 2
      }
    ]
  },
  'week-10': {
    id: 'quiz-week10',
    title: 'LLMs & Prompt Engineering Quiz',
    passingScore: 70,
    questions: [
      {
        id: 'q1',
        question: 'What is tokenization in LLMs?',
        options: [
          'Encrypting data',
          'Breaking text into smaller units (tokens)',
          'Training a model',
          'Visualizing text'
        ],
        correctAnswer: 1
      },
      {
        id: 'q2',
        question: 'What is a hallucination in LLMs?',
        options: [
          'A visual effect',
          'When the model generates false or made-up information',
          'A training technique',
          'A type of architecture'
        ],
        correctAnswer: 1
      },
      {
        id: 'q3',
        question: 'What is chain-of-thought prompting?',
        options: [
          'Linking multiple AI models',
          'Asking AI to reason step-by-step',
          'A database technique',
          'A visualization method'
        ],
        correctAnswer: 1
      },
      {
        id: 'q4',
        question: 'What is the potential annual savings from the LLM Customer Support project?',
        options: [
          '$50,000',
          '$100,000',
          '$500,000+',
          '$1,000,000+'
        ],
        correctAnswer: 2
      },
      {
        id: 'q5',
        question: 'Which company created Claude?',
        options: [
          'OpenAI',
          'Google',
          'Anthropic',
          'Meta'
        ],
        correctAnswer: 2
      }
    ]
  },
  'week-11': {
    id: 'quiz-week11',
    title: 'RAG & Agentic AI Quiz',
    passingScore: 70,
    questions: [
      {
        id: 'q1',
        question: 'What does RAG stand for?',
        options: [
          'Random Access Generation',
          'Retrieval Augmented Generation',
          'Rapid AI Growth',
          'Recursive Algorithm Generator'
        ],
        correctAnswer: 1
      },
      {
        id: 'q2',
        question: 'What is a vector database used for in RAG?',
        options: [
          'Storing images',
          'Storing and searching text embeddings efficiently',
          'Running SQL queries',
          'Training models'
        ],
        correctAnswer: 1
      },
      {
        id: 'q3',
        question: 'What is the typical chunk size for RAG documents?',
        options: [
          '50-100 tokens',
          '500-1000 tokens',
          '5000-10000 tokens',
          'Entire documents'
        ],
        correctAnswer: 1
      },
      {
        id: 'q4',
        question: 'Which is NOT a vector database option?',
        options: [
          'ChromaDB',
          'FAISS',
          'Pinecone',
          'PostgreSQL'
        ],
        correctAnswer: 3
      },
      {
        id: 'q5',
        question: 'What is an AI agent?',
        options: [
          'A human who uses AI',
          'An AI system that can take actions and use tools autonomously',
          'A type of database',
          'A training algorithm'
        ],
        correctAnswer: 1
      }
    ]
  },
  'week-12': {
    id: 'quiz-week12',
    title: 'Capstone & Career Quiz',
    passingScore: 70,
    questions: [
      {
        id: 'q1',
        question: 'What should a good data science portfolio include?',
        options: [
          'Only code files',
          'Projects with clear documentation and business impact',
          'Only certifications',
          'Personal photos'
        ],
        correctAnswer: 1
      },
      {
        id: 'q2',
        question: 'Why is GitHub important for data scientists?',
        options: [
          'Social media presence',
          'Showcasing code, collaboration, and version control',
          'Storing personal files',
          'Watching videos'
        ],
        correctAnswer: 1
      },
      {
        id: 'q3',
        question: 'What is the typical salary range for Junior Data Scientists?',
        options: [
          '$30,000 - $50,000',
          '$88,000 - $110,000',
          '$200,000 - $300,000',
          '$500,000+'
        ],
        correctAnswer: 1
      },
      {
        id: 'q4',
        question: 'What should you emphasize in a data science interview?',
        options: [
          'Memorized definitions only',
          'Problem-solving approach and communication of findings',
          'Speed of typing',
          'Number of certifications'
        ],
        correctAnswer: 1
      },
      {
        id: 'q5',
        question: 'How many portfolio projects does this bootcamp include?',
        options: [
          '5 projects',
          '8 projects',
          '11 projects',
          '15 projects'
        ],
        correctAnswer: 2
      }
    ]
  }
};

// Helper to get all weeks as flat array
export const getAllWeeks = () => {
  const weeks = [{ ...curriculum.preWork, moduleTitle: 'Pre-Work' }];
  curriculum.modules.forEach(module => {
    module.weeks.forEach(week => {
      weeks.push({ ...week, moduleTitle: module.title });
    });
  });
  return weeks;
};

// Get total count
export const getTotalWeeks = () => getAllWeeks().length;

// Get week index in unlock order
export const getWeekIndex = (weekId) => weekUnlockOrder.indexOf(weekId);

// Check if a week should be unlocked based on progress
export const isWeekUnlocked = (weekId, completedQuizzes, completedExercises) => {
  const weekIndex = getWeekIndex(weekId);
  
  // Pre-work and Week 1 are always unlocked
  if (weekIndex <= 1) return true;
  
  // Check if all previous weeks have completed quiz and exercise (if applicable)
  for (let i = 0; i < weekIndex; i++) {
    const prevWeekId = weekUnlockOrder[i];
    
    // Must have completed the quiz for previous week
    if (!completedQuizzes[prevWeekId]) {
      return false;
    }
    
    // Check if previous week has an exercise, and if so, if it's completed
    const prevWeek = getAllWeeks().find(w => w.id === prevWeekId);
    if (prevWeek?.exercise && !completedExercises[prevWeek.exercise.id]) {
      return false;
    }
  }
  
  return true;
};
