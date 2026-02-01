// AI Fundamentals Bootcamp - Curriculum Data (continued)
// This is the complete file - copy the entire content

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

print("Hiring rate by gender:")
print(df.groupby('gender')['hired'].mean())
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
            'Analyze a provided dataset, identify patterns',
            'Document data quality issues found',
            'Create visualizations to communicate insights'
          ],
          resources: [
            { title: 'Pandas Documentation', url: 'https://pandas.pydata.org/docs/', type: 'docs' },
            { title: 'Kaggle - Data Cleaning', url: 'https://www.kaggle.com/learn/data-cleaning', type: 'tutorial' },
            { title: 'Matplotlib Tutorial', url: 'https://matplotlib.org/stable/tutorials/index.html', type: 'tutorial' },
            { title: 'IBM HR Attrition Dataset', url: 'https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset', type: 'tool' }
          ],
          exercise: {
            id: 'ex-week2',
            title: 'EDA on IBM HR Attrition Dataset',
            description: 'Perform data cleaning, visualization, and business recommendations on the IBM HR dataset. Goal: Identify factors contributing to employee attrition.',
            starterCode: `import pandas as pd
import numpy as np

# IBM HR Attrition Dataset (simplified)
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

# TODO: Perform EDA
# 1. Data cleaning - check for missing values, duplicates
# 2. Summary statistics
# 3. Visualize attrition by department, overtime, satisfaction
# 4. Business recommendations to reduce attrition

print("Dataset Shape:", df.shape)
print(df.head())
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

print("=== DATA CLEANING ===")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")

print("\\n=== ATTRITION ANALYSIS ===")
attrition_rate = (df['Attrition'] == 'Yes').mean() * 100
print(f"Overall Attrition Rate: {attrition_rate:.1f}%")

print("\\nAttrition by Department:")
print(df.groupby('Department')['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100).round(1))

print("\\nAttrition by Overtime:")
print(df.groupby('OverTime')['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100).round(1))

print("\\n=== BUSINESS RECOMMENDATIONS ===")
print("1. Review overtime policies - correlates with higher attrition")
print("2. Focus retention efforts on low satisfaction employees")
print("3. Department-specific strategies based on attrition rates")`
          }
        },
        {
          id: 'week-3',
          title: 'Week 3: Python, SQL & Tools Setup',
          sessions: [
            'Python basics—variables, data types, lists, loops, functions (Google Colab environment)',
            'SQL fundamentals—SELECT, WHERE, JOIN, GROUP BY, aggregations; connecting Python to databases'
          ],
          asyncWork: [
            'Complete Python and SQL exercises',
            'Set up working environment (Google Colab, Git/GitHub)',
            'Practice data extraction from SQLite database'
          ],
          sqlTopics: [
            'Basic queries (SELECT, WHERE, ORDER BY)',
            'Aggregations (GROUP BY, COUNT, SUM, AVG)',
            'JOINs (INNER, LEFT)',
            'Subqueries and CTEs',
            'Window functions',
            'Python-database connectivity (sqlite3, pandas.read_sql)'
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
            description: 'Analyze $2.4M in e-commerce transactions. Practice SQL queries including subqueries, CTEs, window functions, customer segmentation, and cohort analysis.',
            starterCode: `import pandas as pd
import numpy as np

np.random.seed(42)
n_transactions = 5000
data = {
    'transaction_id': range(1, n_transactions + 1),
    'customer_id': np.random.randint(1, 501, n_transactions),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports', 'Books'], n_transactions),
    'amount': np.random.uniform(10, 800, n_transactions).round(2),
    'transaction_date': pd.date_range('2025-01-01', periods=n_transactions, freq='H')
}
df = pd.DataFrame(data)

print(f"Total Revenue: \${df['amount'].sum():,.2f}")

# TODO: Write SQL-style queries using pandas
# 1. Revenue by product category (GROUP BY)
# 2. Top 10 customers by total spend
# 3. Monthly revenue trend
# 4. Customer segmentation (High/Medium/Low value)
# 5. Cohort analysis
`,
            solution: `import pandas as pd
import numpy as np

np.random.seed(42)
n_transactions = 5000
data = {
    'transaction_id': range(1, n_transactions + 1),
    'customer_id': np.random.randint(1, 501, n_transactions),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports', 'Books'], n_transactions),
    'amount': np.random.uniform(10, 800, n_transactions).round(2),
    'transaction_date': pd.date_range('2025-01-01', periods=n_transactions, freq='H')
}
df = pd.DataFrame(data)

print(f"Total Revenue: \${df['amount'].sum():,.2f}")

# 1. Revenue by category
print("\\n=== REVENUE BY CATEGORY ===")
print(df.groupby('product_category')['amount'].agg(['sum', 'mean', 'count']).round(2))

# 2. Top 10 customers
print("\\n=== TOP 10 CUSTOMERS ===")
print(df.groupby('customer_id')['amount'].sum().nlargest(10).round(2))

# 3. Monthly trend
df['month'] = df['transaction_date'].dt.to_period('M')
print("\\n=== MONTHLY REVENUE ===")
print(df.groupby('month')['amount'].sum().round(2))

# 4. Customer segmentation
customer_totals = df.groupby('customer_id')['amount'].sum()
def segment(total):
    if total > 2000: return 'High Value'
    elif total > 1000: return 'Medium Value'
    else: return 'Low Value'
print("\\n=== CUSTOMER SEGMENTS ===")
print(customer_totals.apply(segment).value_counts())`
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
            'Build regression models, experiment with feature engineering',
            'Interpret model coefficients for business insights',
            'Document model performance and findings'
          ],
          resources: [
            { title: 'Scikit-learn Linear Models', url: 'https://scikit-learn.org/stable/modules/linear_model.html', type: 'docs' },
            { title: 'StatQuest - Linear Regression', url: 'https://www.youtube.com/watch?v=nk2CQITm_eo', type: 'video' },
            { title: 'Lending Club Dataset', url: 'https://www.kaggle.com/datasets/wordsforthewise/lending-club', type: 'tool' }
          ],
          exercise: {
            id: 'ex-week4',
            title: 'Lending Club Interest Rate Prediction (Part A)',
            description: 'Predict interest rates on 890K+ loans using regression techniques.',
            starterCode: `import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
n = 1000

loan_amount = np.random.randint(1000, 40000, n)
annual_income = np.random.randint(30000, 150000, n)
debt_to_income = np.random.uniform(0, 40, n)
credit_score = np.random.randint(600, 850, n)

interest_rate = 5 + (40000 - loan_amount) * 0.0001 + (850 - credit_score) * 0.02 + debt_to_income * 0.1 + np.random.randn(n) * 2

X = np.column_stack([loan_amount, annual_income, debt_to_income, credit_score])
y = interest_rate

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TODO: Train and evaluate regression model
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

interest_rate = 5 + (40000 - loan_amount) * 0.0001 + (850 - credit_score) * 0.02 + debt_to_income * 0.1 + np.random.randn(n) * 2

X = np.column_stack([loan_amount, annual_income, debt_to_income, credit_score])
y = interest_rate
feature_names = ['loan_amount', 'annual_income', 'debt_to_income', 'credit_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}%")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

print("\\nFeature Coefficients:")
for name, coef in zip(feature_names, model.coef_):
    print(f"  {name}: {coef:.6f}")`
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
            'Classification exercises, compare model approaches',
            'Handle class imbalance techniques (SMOTE, class weights)',
            'Document model selection rationale'
          ],
          resources: [
            { title: 'Scikit-learn Classification', url: 'https://scikit-learn.org/stable/modules/tree.html', type: 'docs' },
            { title: 'StatQuest - Decision Trees', url: 'https://www.youtube.com/watch?v=_L39rN6gz7Y', type: 'video' },
            { title: 'StatQuest - Random Forests', url: 'https://www.youtube.com/watch?v=J4Wdy0Wc_xQ', type: 'video' }
          ],
          exercise: {
            id: 'ex-week5',
            title: 'Lending Club Default Prediction (Part B)',
            description: 'Build a classifier with AUC 0.75+ on imbalanced data. Each prevented default saves ~$9,000.',
            starterCode: `import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

np.random.seed(42)
n = 2000

credit_score = np.random.randint(550, 850, n)
debt_to_income = np.random.uniform(0, 50, n)
loan_amount = np.random.randint(1000, 40000, n)

default_prob = 1 / (1 + np.exp(-(-5 + (700 - credit_score) * 0.02 + debt_to_income * 0.05)))
default = (np.random.random(n) < default_prob).astype(int)

X = np.column_stack([credit_score, debt_to_income, loan_amount])
y = default

print(f"Default rate: {y.mean()*100:.1f}%")

# TODO: Build and evaluate classifier (target AUC > 0.75)
`,
            solution: `import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

np.random.seed(42)
n = 2000

credit_score = np.random.randint(550, 850, n)
debt_to_income = np.random.uniform(0, 50, n)
loan_amount = np.random.randint(1000, 40000, n)

default_prob = 1 / (1 + np.exp(-(-5 + (700 - credit_score) * 0.02 + debt_to_income * 0.05)))
default = (np.random.random(n) < default_prob).astype(int)

X = np.column_stack([credit_score, debt_to_income, loan_amount])
y = default

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
print(f"Random Forest AUC: {rf_auc:.3f}")

print(f"\\nBusiness Impact:")
print(f"Defaults in test: {y_test.sum()}")
print(f"Cost per default: $9,000")
print(f"Potential savings: \${int(y_test.sum() * 0.5 * 9000):,}")`
          }
        },
        {
          id: 'week-6',
          title: 'Week 6: Model Evaluation & Deployment',
          sessions: [
            'Train/test splits, overfitting, cross-validation, comprehensive metrics',
            'Hyperparameter tuning, metrics deep dive, and systematic model selection'
          ],
          asyncWork: [
            'Evaluate models from previous weeks, write up findings',
            'Apply cross-validation techniques',
            'Prepare for deployment (model serialization)',
            'Optional: Time Series Forecasting module (ARIMA, Prophet)'
          ],
          optionalModule: {
            title: 'Time Series Forecasting',
            description: 'Introduction to time series data, trend and seasonality, ARIMA basics, forecasting with Prophet. For finance, supply chain, or demand forecasting roles.'
          },
          resources: [
            { title: 'Scikit-learn Model Evaluation', url: 'https://scikit-learn.org/stable/modules/model_evaluation.html', type: 'docs' },
            { title: 'StatQuest - Cross Validation', url: 'https://www.youtube.com/watch?v=fSytzGwwBVw', type: 'video' },
            { title: 'Streamlit Documentation', url: 'https://docs.streamlit.io/', type: 'docs' }
          ],
          exercise: {
            id: 'ex-week6',
            title: 'Model Evaluation & Streamlit Deployment (Parts C & D)',
            description: 'Complete model evaluation with cross-validation. Part D (Bonus): Deploy as interactive Streamlit web application.',
            starterCode: `import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, weights=[0.85, 0.15], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# TODO: Complete evaluation with confusion matrix, metrics, cross-validation
`,
            solution: `import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score

X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, weights=[0.85, 0.15], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\\n  TN={cm[0,0]} FP={cm[0,1]}\\n  FN={cm[1,0]} TP={cm[1,1]}")
print(f"\\nPrecision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1: {f1_score(y_test, y_pred):.3f}")
print(f"AUC: {roc_auc_score(y_test, y_proba):.3f}")

cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(f"\\n5-Fold CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")`
          }
        },
        {
          id: 'week-7',
          title: 'Week 7: Unsupervised Learning & Recommendations',
          sessions: [
            'Clustering with K-Means, dimensionality reduction with PCA, similarity measures',
            'Recommendation systems—collaborative filtering, content-based approaches, matrix factorization (SVD)'
          ],
          asyncWork: [
            'Clustering exercises, build simple recommender',
            'Handle preprocessing for recommendation systems',
            'Evaluate recommendations with RMSE and precision@k'
          ],
          keyConcepts: [
            'User-item interaction matrices',
            'Cosine similarity',
            'User-based vs. item-based collaborative filtering',
            'Cold start problem',
            'Evaluation metrics (RMSE, precision@k)'
          ],
          resources: [
            { title: 'Scikit-learn Clustering', url: 'https://scikit-learn.org/stable/modules/clustering.html', type: 'docs' },
            { title: 'Google - Recommendation Systems', url: 'https://developers.google.com/machine-learning/recommendation', type: 'course' },
            { title: 'MovieLens Dataset', url: 'https://grouplens.org/datasets/movielens/', type: 'tool' }
          ],
          exercise: {
            id: 'ex-week7',
            title: 'MovieLens Recommendation System',
            description: 'Build a hybrid recommender combining collaborative filtering and content-based approaches on 100K ratings.',
            starterCode: `import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(42)
n_users, n_movies = 100, 50

ratings = np.zeros((n_users, n_movies))
for i in range(n_users):
    rated_movies = np.random.choice(n_movies, np.random.randint(10, 21), replace=False)
    ratings[i, rated_movies] = np.random.randint(1, 6, len(rated_movies))

print(f"Sparsity: {(ratings == 0).mean()*100:.1f}%")

# TODO: Build collaborative filtering recommender
`,
            solution: `import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(42)
n_users, n_movies = 100, 50

ratings = np.zeros((n_users, n_movies))
for i in range(n_users):
    rated_movies = np.random.choice(n_movies, np.random.randint(10, 21), replace=False)
    ratings[i, rated_movies] = np.random.randint(1, 6, len(rated_movies))

# User-user similarity
user_sim = cosine_similarity(ratings)
np.fill_diagonal(user_sim, 0)

# Top-5 recommendations for User 0
target_user = 0
unrated = np.where(ratings[target_user] == 0)[0]
predictions = []
for movie_idx in unrated:
    rated_by = ratings[:, movie_idx] > 0
    if rated_by.sum() > 0:
        sims = user_sim[target_user, rated_by]
        if np.abs(sims).sum() > 0:
            pred = np.dot(sims, ratings[rated_by, movie_idx]) / np.abs(sims).sum()
            predictions.append((f'Movie_{movie_idx}', pred))

predictions.sort(key=lambda x: x[1], reverse=True)
print("Top 5 Recommendations:")
for movie, score in predictions[:5]:
    print(f"  {movie}: {score:.2f}")`
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
            '★ Capstone Project: Receive guidelines and begin project planning'
          ],
          capstoneIntro: {
            description: 'Students receive capstone guidelines and begin project planning',
            requirements: [
              'Address a real-world problem using data science/ML techniques',
              'Include data collection/preparation, analysis, and modeling',
              'Demonstrate at least two skills from the curriculum',
              'Show clear business impact or practical application',
              'Include ethical considerations relevant to the project'
            ],
            deliverables: [
              'Written report explaining problem, approach, methodology, findings',
              'Working code/prototype hosted on GitHub with documentation',
              '10-minute presentation demonstrating solution and key insights'
            ],
            timeline: 'Four weeks to complete (Weeks 9-12)'
          },
          resources: [
            { title: 'TensorFlow Tutorials', url: 'https://www.tensorflow.org/tutorials', type: 'tutorial' },
            { title: 'Keras Documentation', url: 'https://keras.io/guides/', type: 'docs' },
            { title: '3Blue1Brown - Neural Networks', url: 'https://www.youtube.com/watch?v=aircAruvnKk', type: 'video' },
            { title: 'UCI Heart Disease Dataset', url: 'https://archive.ics.uci.edu/dataset/45/heart+disease', type: 'tool' }
          ],
          exercise: {
            id: 'ex-week8',
            title: 'Heart Disease Prediction with Neural Networks',
            description: 'Build a neural network using the UCI dataset. Compare NN vs traditional ML. Business impact: ~$75K per missed diagnosis.',
            starterCode: `import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

np.random.seed(42)
n = 500

age = np.random.randint(30, 80, n)
cholesterol = np.random.randint(150, 400, n)
max_heart_rate = np.random.randint(80, 200, n)

disease_prob = 1 / (1 + np.exp(-(-3 + age * 0.04 + cholesterol * 0.003 - max_heart_rate * 0.02)))
heart_disease = (np.random.random(n) < disease_prob).astype(int)

X = np.column_stack([age, cholesterol, max_heart_rate])
y = heart_disease

print(f"Disease prevalence: {y.mean()*100:.1f}%")
print(f"Business context: ~$75K cost per missed diagnosis")

# TODO: Scale features, train baseline, design NN architecture
`,
            solution: `import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

np.random.seed(42)
n = 500

age = np.random.randint(30, 80, n)
cholesterol = np.random.randint(150, 400, n)
max_heart_rate = np.random.randint(80, 200, n)

disease_prob = 1 / (1 + np.exp(-(-3 + age * 0.04 + cholesterol * 0.003 - max_heart_rate * 0.02)))
heart_disease = (np.random.random(n) < disease_prob).astype(int)

X = np.column_stack([age, cholesterol, max_heart_rate])
y = heart_disease

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print(f"Logistic Regression AUC: {roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1]):.3f}")

print("\\nNeural Network Architecture:")
print("  Input: 3 features")
print("  Hidden 1: 16 neurons, ReLU")
print("  Hidden 2: 8 neurons, ReLU")
print("  Output: 1 neuron, Sigmoid")`
          }
        },
        {
          id: 'week-9',
          title: 'Week 9: Deep Learning Applications',
          sessions: [
            'Convolutional neural networks for images, transfer learning with pre-trained models (VGG16, ResNet, MobileNet)',
            'Sequence models and recurrent networks (RNNs, LSTMs), introduction to attention mechanisms and transformers'
          ],
          asyncWork: [
            'Image classification mini-project using transfer learning',
            'Readings on transformer architecture',
            'Continue capstone project'
          ],
          keyBridge: 'Week 9 connects traditional deep learning to modern LLMs by explaining why Transformers replaced RNNs (parallelization, direct long-range connections)',
          resources: [
            { title: 'TensorFlow Transfer Learning', url: 'https://www.tensorflow.org/tutorials/images/transfer_learning', type: 'tutorial' },
            { title: 'Illustrated Transformer', url: 'https://jalammar.github.io/illustrated-transformer/', type: 'article' },
            { title: 'Kaggle Image Classification', url: 'https://www.kaggle.com/competitions?search=image+classification', type: 'tool' }
          ],
          exercise: {
            id: 'ex-week9',
            title: 'Image Classification with Transfer Learning',
            description: 'Use pre-trained models (VGG16/ResNet) on a Kaggle dataset. Demonstrate 80%+ reduction in training time.',
            starterCode: `# Transfer Learning for Image Classification
print("TRANSFER LEARNING WORKFLOW")
print("1. Load pre-trained model (VGG16/ResNet)")
print("2. Freeze base layers")
print("3. Add custom classification head")
print("4. Train on your dataset")

# TODO: Explain why this works and when to use it
`,
            solution: `print("TRANSFER LEARNING IMPLEMENTATION")
print("""
# Keras code:
base_model = VGG16(weights='imagenet', include_top=False)
for layer in base_model.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)
""")

print("\\nTraining Time Comparison:")
print("  From scratch: 100 epochs, 10+ hours")
print("  Transfer: 10 epochs, <1 hour (80%+ reduction)")

print("\\nWhy Transformers Replaced RNNs:")
print("  - Parallel processing (faster)")
print("  - Direct long-range connections")
print("  - No vanishing gradients")`
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
            'How LLMs work—tokenization (BPE algorithm), embeddings (semantic similarity), next-token prediction, temperature, scaling laws, LLM landscape (GPT, Claude, LLaMA, Gemini, Mistral)',
            'Capabilities and limitations (hallucinations, knowledge cutoff, reasoning gaps, context limits), prompt engineering techniques (zero-shot, few-shot, chain-of-thought), working with LLM APIs (OpenAI, Anthropic)'
          ],
          asyncWork: [
            'LLM fundamentals review, prompt engineering practice exercises',
            'API exploration and experimentation (OpenAI, Anthropic)',
            'Continue capstone project'
          ],
          keyConcepts: [
            'Tokenization and token efficiency (cost optimization)',
            'Text embeddings and semantic similarity',
            'LLM training pipeline: pre-training → SFT → RLHF',
            'Prompt engineering: zero-shot, few-shot, chain-of-thought',
            'Full prompt template: ROLE → CONTEXT → TASK → FORMAT → CONSTRAINTS',
            'LLM APIs (OpenAI, Anthropic) with token tracking and error handling',
            'Understanding and mitigating hallucinations (RAG preview)'
          ],
          resources: [
            { title: 'Anthropic - Claude Documentation', url: 'https://docs.anthropic.com/', type: 'docs' },
            { title: 'OpenAI - GPT Best Practices', url: 'https://platform.openai.com/docs/guides/prompt-engineering', type: 'docs' },
            { title: 'Prompt Engineering Guide', url: 'https://www.promptingguide.ai/', type: 'tutorial' }
          ],
          exercise: {
            id: 'ex-week10',
            title: 'LLM-Powered Customer Support System',
            description: 'Build ticket classification (85%+ accuracy) and response generation using prompt engineering. Potential: $500K+ annual savings.',
            starterCode: `# LLM Customer Support System
tickets = [
    {"id": 1, "text": "I can't log into my account"},
    {"id": 2, "text": "When will my order arrive?"},
    {"id": 3, "text": "I want a refund"},
]
categories = ["Account Issues", "Shipping", "Refunds", "Billing", "Technical Support"]

# TODO: Design prompts for classification and response generation
`,
            solution: `print("ZERO-SHOT PROMPT:")
print('Classify this ticket: "{text}" into one category: Account Issues, Shipping, Refunds, Billing, Technical Support')

print("\\nFEW-SHOT PROMPT:")
print('Examples: "I forgot my password" → Account Issues, "Where is my package?" → Shipping')

print("\\nRESPONSE GENERATION (ROLE → CONTEXT → TASK → FORMAT → CONSTRAINTS):")
print("ROLE: Customer support agent")
print("CONTEXT: Ticket: {text}, Category: {category}")
print("TASK: Write helpful response")
print("FORMAT: Greeting, acknowledgment, solution, closing")
print("CONSTRAINTS: Under 100 words, empathetic, no timeline promises")

print("\\nBUSINESS IMPACT: $500K+ annual savings with 85% auto-resolution")`
          }
        },
        {
          id: 'week-11',
          title: 'Week 11: RAG & Agentic AI',
          sessions: [
            'Embeddings deep dive—text embeddings at scale, vector similarity search, vector databases (ChromaDB, FAISS, Pinecone), document chunking strategies',
            'RAG architecture—retrieval, augmentation, generation pipeline; building RAG applications with LangChain; introduction to Agentic AI concepts (tool use, function calling, autonomous agents)'
          ],
          asyncWork: [
            'Build a RAG-powered application using LangChain and a vector database',
            'Implement document Q&A system',
            'Continue capstone project'
          ],
          keyConcepts: [
            'Text embeddings and vector similarity search at scale',
            'Vector databases: ChromaDB, FAISS, Pinecone comparison',
            'Chunking strategies for document processing',
            'RAG architecture: retrieval → augmentation → generation',
            'LangChain fundamentals for building LLM applications',
            'Agentic AI concepts: tool use, function calling, autonomous agents (overview)'
          ],
          resources: [
            { title: 'LangChain Documentation', url: 'https://python.langchain.com/docs/', type: 'docs' },
            { title: 'Pinecone - Vector DB Guide', url: 'https://www.pinecone.io/learn/', type: 'tutorial' },
            { title: 'ChromaDB', url: 'https://docs.trychroma.com/', type: 'docs' }
          ],
          exercise: {
            id: 'ex-week11',
            title: 'Document Q&A System',
            description: 'Build a RAG-powered company knowledge base that answers questions about uploaded documents.',
            starterCode: `# RAG Document Q&A System
documents = [
    {"title": "Return Policy", "content": "Items can be returned within 30 days..."},
    {"title": "Shipping Info", "content": "Standard shipping takes 5-7 business days..."},
]
question = "How long do I have to return an item?"

# TODO: Build RAG pipeline (chunk, embed, store, retrieve, augment, generate)
`,
            solution: `print("RAG PIPELINE:")
print("1. CHUNK: Split documents into 500-1000 token chunks")
print("2. EMBED: Convert to vectors with sentence-transformers")
print("3. STORE: Save in ChromaDB/FAISS/Pinecone")
print("4. RETRIEVE: Find top-k similar chunks for query")
print("5. AUGMENT: Add retrieved context to prompt")
print("6. GENERATE: LLM answers based on context only")

print("\\nAugmented Prompt:")
print('Context: "Items can be returned within 30 days..."')
print('Question: "How long to return?"')
print('Answer: "You have 30 days to return an item."')

print("\\nVector DB Comparison:")
print("  ChromaDB: Easy setup, local, free")
print("  FAISS: Fast, Facebook, large scale")
print("  Pinecone: Cloud, managed, paid")`
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
          officeHours: 'Extended session for final Q&A, career paths in AI, continued learning resources, portfolio review',
          resources: [
            { title: 'GitHub Profile Guide', url: 'https://docs.github.com/en/account-and-profile', type: 'article' },
            { title: 'Data Science Portfolio Tips', url: 'https://www.datacamp.com/blog/how-to-build-a-data-science-portfolio', type: 'article' },
            { title: 'Kaggle Competitions', url: 'https://www.kaggle.com/competitions', type: 'tool' }
          ],
          exercise: null
        }
      ]
    }
  ]
};

// Week unlock order for progressive learning
export const weekUnlockOrder = [
  'pre-work',
  'week-1',
  'week-2',
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
      { id: 'q1', question: 'What is algorithmic bias?', options: ['A type of computer virus', 'Systematic errors in AI systems that create unfair outcomes', 'A programming language', 'A hardware malfunction'], correctAnswer: 1 },
      { id: 'q2', question: 'Which is a key principle of responsible AI?', options: ['Maximizing profit at all costs', 'Hiding how AI systems make decisions', 'Transparency and explainability', 'Using data without consent'], correctAnswer: 2 },
      { id: 'q3', question: 'What is differential privacy?', options: ['A technique to protect individual data while allowing useful analysis', 'A way to make AI faster', 'A type of neural network', 'A programming framework'], correctAnswer: 0 },
      { id: 'q4', question: 'Why is AI fairness important?', options: ['It makes code run faster', 'It ensures AI systems do not discriminate', 'It reduces costs', 'It is not important'], correctAnswer: 1 },
      { id: 'q5', question: 'What should you do if you discover bias in your AI model?', options: ['Ignore it', 'Investigate, mitigate, and document', 'Delete the data', 'Blame others'], correctAnswer: 1 }
    ]
  },
  'week-1': {
    id: 'quiz-week1',
    title: 'What Is AI? Quiz',
    passingScore: 70,
    questions: [
      { id: 'q1', question: 'What is Machine Learning?', options: ['A type of robot', 'A subset of AI where systems learn from data', 'A programming language', 'A database'], correctAnswer: 1 },
      { id: 'q2', question: 'Which is NOT a type of machine learning?', options: ['Supervised learning', 'Unsupervised learning', 'Reinforcement learning', 'Mechanical learning'], correctAnswer: 3 },
      { id: 'q3', question: 'What is Deep Learning?', options: ['Learning while sleeping', 'ML using neural networks with many layers', 'A database query', 'A hardware component'], correctAnswer: 1 },
      { id: 'q4', question: 'What does "training" mean in ML?', options: ['Teaching humans', 'A model learning patterns from data', 'Installing software', 'Writing docs'], correctAnswer: 1 },
      { id: 'q5', question: 'Which company created ChatGPT?', options: ['Google', 'Meta', 'OpenAI', 'Microsoft'], correctAnswer: 2 }
    ]
  },
  'week-2': {
    id: 'quiz-week2',
    title: 'Data Thinking Quiz',
    passingScore: 70,
    questions: [
      { id: 'q1', question: 'What is EDA?', options: ['A programming language', 'Analyzing data sets to summarize characteristics', 'A database', 'A ML model'], correctAnswer: 1 },
      { id: 'q2', question: 'Which is NOT a data quality issue?', options: ['Missing values', 'Duplicates', 'Data being too accurate', 'Inconsistent formatting'], correctAnswer: 2 },
      { id: 'q3', question: 'What does the median represent?', options: ['The average', 'The most frequent value', 'The middle value when sorted', 'The range'], correctAnswer: 2 },
      { id: 'q4', question: 'Why is data visualization important?', options: ['Makes reports longer', 'Helps identify patterns', 'Slows analysis', 'Not important'], correctAnswer: 1 },
      { id: 'q5', question: 'What dataset is used in the Week 2 project?', options: ['MovieLens', 'Lending Club', 'IBM HR Attrition', 'MNIST'], correctAnswer: 2 }
    ]
  },
  'week-3': {
    id: 'quiz-week3',
    title: 'Python & SQL Quiz',
    passingScore: 70,
    questions: [
      { id: 'q1', question: 'What does SQL stand for?', options: ['Simple Question Language', 'Structured Query Language', 'System Quality Level', 'Standard Query Logic'], correctAnswer: 1 },
      { id: 'q2', question: 'Which SQL clause filters results?', options: ['SELECT', 'FROM', 'WHERE', 'ORDER BY'], correctAnswer: 2 },
      { id: 'q3', question: 'What does a JOIN do?', options: ['Deletes tables', 'Combines rows from tables', 'Creates a database', 'Sorts data'], correctAnswer: 1 },
      { id: 'q4', question: 'Which Python structure uses key-value pairs?', options: ['List', 'Tuple', 'Dictionary', 'Set'], correctAnswer: 2 },
      { id: 'q5', question: 'What is the TechMart project transaction value?', options: ['$500K', '$1.2M', '$2.4M', '$5M'], correctAnswer: 2 }
    ]
  },
  'week-4': {
    id: 'quiz-week4',
    title: 'Regression Quiz',
    passingScore: 70,
    questions: [
      { id: 'q1', question: 'What is linear regression used for?', options: ['Classification', 'Predicting continuous values', 'Clustering', 'Image recognition'], correctAnswer: 1 },
      { id: 'q2', question: 'What is a "feature" in ML?', options: ['The output', 'An input variable for prediction', 'An algorithm', 'A visualization'], correctAnswer: 1 },
      { id: 'q3', question: 'What metric evaluates regression models?', options: ['Accuracy', 'RMSE', 'F1 Score', 'Precision'], correctAnswer: 1 },
      { id: 'q4', question: 'What does the Lending Club Part A predict?', options: ['Defaults', 'Interest rates', 'Customer satisfaction', 'Stock prices'], correctAnswer: 1 },
      { id: 'q5', question: 'What does a negative coefficient mean?', options: ['Model failed', 'As feature increases, target decreases', 'Feature not important', 'Error'], correctAnswer: 1 }
    ]
  },
  'week-5': {
    id: 'quiz-week5',
    title: 'Classification Quiz',
    passingScore: 70,
    questions: [
      { id: 'q1', question: 'What is classification for?', options: ['Predicting continuous values', 'Predicting categorical outcomes', 'Grouping data', 'Reducing dimensions'], correctAnswer: 1 },
      { id: 'q2', question: 'What is a Random Forest?', options: ['A single tree', 'An ensemble of decision trees', 'A data type', 'A visualization'], correctAnswer: 1 },
      { id: 'q3', question: 'What does AUC-ROC measure?', options: ['Speed', 'Classification performance across thresholds', 'Data size', 'Feature importance'], correctAnswer: 1 },
      { id: 'q4', question: 'How much does each prevented default save?', options: ['$1,000', '$5,000', '$9,000', '$50,000'], correctAnswer: 2 },
      { id: 'q5', question: 'What is class imbalance?', options: ['Different feature scales', 'One class much more frequent', 'Complex models', 'Missing data'], correctAnswer: 1 }
    ]
  },
  'week-6': {
    id: 'quiz-week6',
    title: 'Model Evaluation Quiz',
    passingScore: 70,
    questions: [
      { id: 'q1', question: 'What is overfitting?', options: ['Good performance on new data', 'Learning noise, poor on new data', 'Too simple model', 'Long training'], correctAnswer: 1 },
      { id: 'q2', question: 'What is cross-validation for?', options: ['Cleaning data', 'Evaluating model robustly', 'Visualization', 'Faster training'], correctAnswer: 1 },
      { id: 'q3', question: 'When prioritize recall over precision?', options: ['False positives costly', 'False negatives costly', 'Always', 'Never'], correctAnswer: 1 },
      { id: 'q4', question: 'What is Streamlit for?', options: ['Training models', 'Building web applications', 'Data cleaning', 'Database management'], correctAnswer: 1 },
      { id: 'q5', question: 'What does a confusion matrix show?', options: ['Parameters', 'Predictions vs actuals (TP, TN, FP, FN)', 'Training time', 'Correlations'], correctAnswer: 1 }
    ]
  },
  'week-7': {
    id: 'quiz-week7',
    title: 'Unsupervised & Recommendations Quiz',
    passingScore: 70,
    questions: [
      { id: 'q1', question: 'What is clustering?', options: ['Predicting labels', 'Grouping similar data points', 'Training NNs', 'Visualization'], correctAnswer: 1 },
      { id: 'q2', question: 'What is collaborative filtering?', options: ['Filtering spam', 'Recommending based on similar users', 'Cleaning data', 'A clustering method'], correctAnswer: 1 },
      { id: 'q3', question: 'What is the cold start problem?', options: ['Servers too cold', 'Difficulty with new users/items with no history', 'Slow computation', 'Storage issues'], correctAnswer: 1 },
      { id: 'q4', question: 'What dataset is used in Week 7?', options: ['Lending Club', 'MovieLens 100K', 'IBM HR', 'MNIST'], correctAnswer: 1 },
      { id: 'q5', question: 'What does PCA stand for?', options: ['Primary Component Analysis', 'Principal Component Analysis', 'Partial Cluster Algorithm', 'Predictive Classification'], correctAnswer: 1 }
    ]
  },
  'week-8': {
    id: 'quiz-week8',
    title: 'Neural Networks Quiz',
    passingScore: 70,
    questions: [
      { id: 'q1', question: 'What is an activation function?', options: ['Starts training', 'Introduces non-linearity', 'Loads data', 'Calculates loss'], correctAnswer: 1 },
      { id: 'q2', question: 'What is backpropagation?', options: ['Moving data backwards', 'Calculating gradients to update weights', 'A NN type', 'Preprocessing'], correctAnswer: 1 },
      { id: 'q3', question: 'Cost of missed heart disease diagnosis?', options: ['$5K', '$25K', '$75K', '$500K'], correctAnswer: 2 },
      { id: 'q4', question: 'When choose NN over logistic regression?', options: ['Small data, linear', 'Large data, complex patterns', 'Always', 'Never'], correctAnswer: 1 },
      { id: 'q5', question: 'What does ReLU do?', options: ['Returns input if positive, else 0', 'Squashes to 0-1', 'Normalizes', 'Calculates loss'], correctAnswer: 0 }
    ]
  },
  'week-9': {
    id: 'quiz-week9',
    title: 'Deep Learning Applications Quiz',
    passingScore: 70,
    questions: [
      { id: 'q1', question: 'What is a CNN for?', options: ['Text', 'Image/visual data', 'Time series', 'Recommendations'], correctAnswer: 1 },
      { id: 'q2', question: 'What is transfer learning?', options: ['Moving data', 'Using pre-trained model for new task', 'Data augmentation', 'File transfer'], correctAnswer: 1 },
      { id: 'q3', question: 'Why freeze base layers?', options: ['Faster training, preserve features', 'Increase model size', 'Add more data', 'Visualization'], correctAnswer: 0 },
      { id: 'q4', question: 'What did Transformers replace RNNs with?', options: ['Convolutions', 'Self-attention', 'Pooling', 'Dropout'], correctAnswer: 1 },
      { id: 'q5', question: 'Transfer learning training time reduction?', options: ['10-20%', '30-50%', '80-90%', 'No reduction'], correctAnswer: 2 }
    ]
  },
  'week-10': {
    id: 'quiz-week10',
    title: 'LLMs & Prompt Engineering Quiz',
    passingScore: 70,
    questions: [
      { id: 'q1', question: 'What is tokenization?', options: ['Encrypting data', 'Breaking text into smaller units', 'Training', 'Visualization'], correctAnswer: 1 },
      { id: 'q2', question: 'What is an LLM hallucination?', options: ['Visual effect', 'Generating false information', 'Training technique', 'Architecture type'], correctAnswer: 1 },
      { id: 'q3', question: 'What is chain-of-thought prompting?', options: ['Linking models', 'Asking AI to reason step-by-step', 'Database technique', 'Visualization'], correctAnswer: 1 },
      { id: 'q4', question: 'LLM Customer Support potential savings?', options: ['$50K', '$100K', '$500K+', '$1M+'], correctAnswer: 2 },
      { id: 'q5', question: 'Who created Claude?', options: ['OpenAI', 'Google', 'Anthropic', 'Meta'], correctAnswer: 2 }
    ]
  },
  'week-11': {
    id: 'quiz-week11',
    title: 'RAG & Agentic AI Quiz',
    passingScore: 70,
    questions: [
      { id: 'q1', question: 'What does RAG stand for?', options: ['Random Access Generation', 'Retrieval Augmented Generation', 'Rapid AI Growth', 'Recursive Algorithm'], correctAnswer: 1 },
      { id: 'q2', question: 'What is a vector database for in RAG?', options: ['Storing images', 'Storing/searching text embeddings', 'SQL queries', 'Training'], correctAnswer: 1 },
      { id: 'q3', question: 'Typical chunk size for RAG?', options: ['50-100', '500-1000', '5000-10000', 'Entire documents'], correctAnswer: 1 },
      { id: 'q4', question: 'Which is NOT a vector database?', options: ['ChromaDB', 'FAISS', 'Pinecone', 'PostgreSQL'], correctAnswer: 3 },
      { id: 'q5', question: 'What is an AI agent?', options: ['Human using AI', 'AI that takes actions/uses tools autonomously', 'A database', 'An algorithm'], correctAnswer: 1 }
    ]
  },
  'week-12': {
    id: 'quiz-week12',
    title: 'Capstone & Career Quiz',
    passingScore: 70,
    questions: [
      { id: 'q1', question: 'What should a good portfolio include?', options: ['Only code', 'Projects with documentation and business impact', 'Only certifications', 'Personal photos'], correctAnswer: 1 },
      { id: 'q2', question: 'Why is GitHub important?', options: ['Social media', 'Showcasing code, collaboration, version control', 'Storing files', 'Watching videos'], correctAnswer: 1 },
      { id: 'q3', question: 'Junior Data Scientist salary range?', options: ['$30K-$50K', '$88K-$110K', '$200K-$300K', '$500K+'], correctAnswer: 1 },
      { id: 'q4', question: 'What to emphasize in interviews?', options: ['Memorized definitions', 'Problem-solving and communication', 'Typing speed', 'Certifications count'], correctAnswer: 1 },
      { id: 'q5', question: 'How many portfolio projects in this bootcamp?', options: ['5', '8', '11', '15'], correctAnswer: 2 }
    ]
  }
};

// Helper functions
export const getAllWeeks = () => {
  const weeks = [{ ...curriculum.preWork, moduleTitle: 'Pre-Work' }];
  curriculum.modules.forEach(module => {
    module.weeks.forEach(week => {
      weeks.push({ ...week, moduleTitle: module.title });
    });
  });
  return weeks;
};

export const getTotalWeeks = () => getAllWeeks().length;

export const getWeekIndex = (weekId) => weekUnlockOrder.indexOf(weekId);

export const isWeekUnlocked = (weekId, completedQuizzes, completedExercises) => {
  const weekIndex = getWeekIndex(weekId);
  if (weekIndex <= 1) return true; // Pre-work and Week 1 always unlocked
  
  for (let i = 0; i < weekIndex; i++) {
    const prevWeekId = weekUnlockOrder[i];
    if (!completedQuizzes[prevWeekId]) return false;
    
    const prevWeek = getAllWeeks().find(w => w.id === prevWeekId);
    if (prevWeek?.exercise && !completedExercises[prevWeek.exercise.id]) return false;
  }
  return true;
};
