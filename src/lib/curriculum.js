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
      title: 'Bias Detection Exercise',
      description: 'Analyze a dataset for potential biases',
      starterCode: `import pandas as pd
import numpy as np

# Sample hiring dataset
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

# Observation: Younger candidates appear to be hired more often
# This could indicate age bias in the hiring process`
    }
  },
  modules: [
    {
      id: 'module-1',
      title: 'Module 1: Foundations',
      weeks: [
        {
          id: 'week-1',
          title: 'Week 1: What Is AI?',
          sessions: [
            'History and landscape of AI, ML, deep learning, generative AI',
            'How machines "learn" from data (intuition-focused)'
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
            'Analyze a provided dataset',
            'Identify patterns and anomalies',
            'Document data quality issues'
          ],
          resources: [
            { title: 'Pandas Documentation', url: 'https://pandas.pydata.org/docs/', type: 'docs' },
            { title: 'Kaggle - Data Cleaning', url: 'https://www.kaggle.com/learn/data-cleaning', type: 'tutorial' },
            { title: 'Matplotlib Tutorial', url: 'https://matplotlib.org/stable/tutorials/index.html', type: 'tutorial' }
          ],
          exercise: {
            id: 'ex-week2',
            title: 'Exploratory Data Analysis',
            description: 'Perform EDA on a sample dataset',
            starterCode: `import pandas as pd
import numpy as np

# Sample sales dataset
np.random.seed(42)
data = {
    'date': pd.date_range('2025-01-01', periods=100),
    'product': np.random.choice(['A', 'B', 'C'], 100),
    'sales': np.random.randint(100, 1000, 100),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
}
df = pd.DataFrame(data)

# TODO: Perform EDA
# 1. Show basic statistics
print(df.describe())

# 2. Check for missing values
# Your code here

# 3. Sales by product
# Your code here

# 4. Sales by region
# Your code here
`,
            solution: `import pandas as pd
import numpy as np

np.random.seed(42)
data = {
    'date': pd.date_range('2025-01-01', periods=100),
    'product': np.random.choice(['A', 'B', 'C'], 100),
    'sales': np.random.randint(100, 1000, 100),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
}
df = pd.DataFrame(data)

# 1. Basic statistics
print("Basic Statistics:")
print(df.describe())

# 2. Missing values
print("\\nMissing Values:")
print(df.isnull().sum())

# 3. Sales by product
print("\\nSales by Product:")
print(df.groupby('product')['sales'].agg(['mean', 'sum', 'count']))

# 4. Sales by region
print("\\nSales by Region:")
print(df.groupby('region')['sales'].agg(['mean', 'sum', 'count']))

# 5. Top selling days
print("\\nTop 5 Sales Days:")
print(df.nlargest(5, 'sales')[['date', 'product', 'sales', 'region']])`
          }
        },
        {
          id: 'week-3',
          title: 'Week 3: Python, SQL & Tools Setup',
          sessions: [
            'Python basics—variables, data types, lists, loops, functions',
            'SQL fundamentals—SELECT, WHERE, JOIN, GROUP BY, aggregations'
          ],
          asyncWork: [
            'Complete Python exercises',
            'Complete SQL exercises',
            'Set up working environment',
            'Practice data extraction from SQLite database'
          ],
          resources: [
            { title: 'Python Official Tutorial', url: 'https://docs.python.org/3/tutorial/', type: 'docs' },
            { title: 'SQLBolt - Interactive SQL', url: 'https://sqlbolt.com/', type: 'tutorial' },
            { title: 'Mode SQL Tutorial', url: 'https://mode.com/sql-tutorial/', type: 'tutorial' },
            { title: 'Google Colab', url: 'https://colab.research.google.com/', type: 'tool' }
          ],
          exercise: {
            id: 'ex-week3',
            title: 'Python & SQL Fundamentals',
            description: 'Practice basic Python and SQL operations',
            starterCode: `# Python Exercise: Data Processing
data = [
    {'name': 'Alice', 'age': 28, 'dept': 'Engineering'},
    {'name': 'Bob', 'age': 35, 'dept': 'Marketing'},
    {'name': 'Carol', 'age': 42, 'dept': 'Engineering'},
    {'name': 'David', 'age': 31, 'dept': 'Sales'},
    {'name': 'Eve', 'age': 29, 'dept': 'Engineering'}
]

# TODO: Complete these exercises

# 1. Calculate average age
# Your code here

# 2. Filter employees in Engineering
# Your code here

# 3. Create a function to find employees by department
def find_by_dept(employees, department):
    # Your code here
    pass

# 4. Sort employees by age
# Your code here
`,
            solution: `data = [
    {'name': 'Alice', 'age': 28, 'dept': 'Engineering'},
    {'name': 'Bob', 'age': 35, 'dept': 'Marketing'},
    {'name': 'Carol', 'age': 42, 'dept': 'Engineering'},
    {'name': 'David', 'age': 31, 'dept': 'Sales'},
    {'name': 'Eve', 'age': 29, 'dept': 'Engineering'}
]

# 1. Calculate average age
avg_age = sum(emp['age'] for emp in data) / len(data)
print(f"Average age: {avg_age}")

# 2. Filter employees in Engineering
engineers = [emp for emp in data if emp['dept'] == 'Engineering']
print(f"Engineers: {engineers}")

# 3. Function to find by department
def find_by_dept(employees, department):
    return [emp for emp in employees if emp['dept'] == department]

print(f"Marketing: {find_by_dept(data, 'Marketing')}")

# 4. Sort by age
sorted_by_age = sorted(data, key=lambda x: x['age'])
print(f"Sorted by age: {sorted_by_age}")`
          }
        }
      ]
    },
    {
      id: 'module-2',
      title: 'Module 2: Core Machine Learning',
      weeks: [
        {
          id: 'week-4',
          title: 'Week 4: Supervised Learning I',
          sessions: [
            'The prediction problem, features vs. targets, regression intuition',
            'Linear regression hands-on, interpreting coefficients'
          ],
          asyncWork: [
            'Build a simple regression model on provided dataset',
            'Interpret model coefficients',
            'Document findings'
          ],
          resources: [
            { title: 'Scikit-learn Linear Models', url: 'https://scikit-learn.org/stable/modules/linear_model.html', type: 'docs' },
            { title: 'StatQuest - Linear Regression', url: 'https://www.youtube.com/watch?v=nk2CQITm_eo', type: 'video' },
            { title: 'Kaggle - Intro to ML', url: 'https://www.kaggle.com/learn/intro-to-machine-learning', type: 'course' }
          ],
          exercise: {
            id: 'ex-week4',
            title: 'Linear Regression: House Prices',
            description: 'Build a linear regression model to predict house prices',
            starterCode: `import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample house data
np.random.seed(42)
n = 100
sqft = np.random.randint(800, 3000, n)
bedrooms = np.random.randint(1, 6, n)
age = np.random.randint(0, 50, n)
price = 50000 + sqft * 150 + bedrooms * 10000 - age * 1000 + np.random.randn(n) * 20000

X = np.column_stack([sqft, bedrooms, age])
y = price

# TODO: Complete the regression analysis
# 1. Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Create and train the model
model = LinearRegression()
# Your code here

# 3. Make predictions
# Your code here

# 4. Evaluate the model (RMSE and R2)
# Your code here
`,
            solution: `import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
n = 100
sqft = np.random.randint(800, 3000, n)
bedrooms = np.random.randint(1, 6, n)
age = np.random.randint(0, 50, n)
price = 50000 + sqft * 150 + bedrooms * 10000 - age * 1000 + np.random.randn(n) * 20000

X = np.column_stack([sqft, bedrooms, age])
y = price

# 1. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 3. Make predictions
y_pred = model.predict(X_test)

# 4. Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: \${rmse:,.2f}")
print(f"R² Score: {r2:.4f}")`
          }
        },
        {
          id: 'week-5',
          title: 'Week 5: Supervised Learning II',
          sessions: [
            'Classification problems, logistic regression, decision boundaries',
            'Decision trees, random forests (intuition and application)'
          ],
          asyncWork: [
            'Classification exercise',
            'Compare model approaches',
            'Document model selection rationale'
          ],
          resources: [
            { title: 'Scikit-learn Classification', url: 'https://scikit-learn.org/stable/modules/tree.html', type: 'docs' },
            { title: 'StatQuest - Decision Trees', url: 'https://www.youtube.com/watch?v=_L39rN6gz7Y', type: 'video' },
            { title: 'StatQuest - Random Forests', url: 'https://www.youtube.com/watch?v=J4Wdy0Wc_xQ', type: 'video' }
          ],
          exercise: {
            id: 'ex-week5',
            title: 'Classification: Customer Churn',
            description: 'Build classification models to predict customer churn',
            starterCode: `import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample customer data
np.random.seed(42)
n = 200
tenure = np.random.randint(1, 72, n)
monthly_charges = np.random.uniform(20, 100, n)
support_calls = np.random.randint(0, 10, n)
churn_prob = 1 / (1 + np.exp(-(support_calls * 0.5 - tenure * 0.05 + monthly_charges * 0.02 - 2)))
churn = (np.random.random(n) < churn_prob).astype(int)

X = np.column_stack([tenure, monthly_charges, support_calls])
y = churn

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TODO: Build and compare models
# 1. Logistic Regression
log_reg = LogisticRegression()
# Your code here

# 2. Decision Tree
dt = DecisionTreeClassifier(max_depth=5)
# Your code here

# 3. Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5)
# Your code here
`,
            solution: `import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)
n = 200
tenure = np.random.randint(1, 72, n)
monthly_charges = np.random.uniform(20, 100, n)
support_calls = np.random.randint(0, 10, n)
churn_prob = 1 / (1 + np.exp(-(support_calls * 0.5 - tenure * 0.05 + monthly_charges * 0.02 - 2)))
churn = (np.random.random(n) < churn_prob).astype(int)

X = np.column_stack([tenure, monthly_charges, support_calls])
y = churn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train all models
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# Compare
print("Model Comparison:")
print(f"Logistic Regression: {accuracy_score(y_test, log_reg.predict(X_test)):.3f}")
print(f"Decision Tree: {accuracy_score(y_test, dt.predict(X_test)):.3f}")
print(f"Random Forest: {accuracy_score(y_test, rf.predict(X_test)):.3f}")`
          }
        },
        {
          id: 'week-6',
          title: 'Week 6: Model Evaluation & Improvement',
          sessions: [
            'Train/test splits, overfitting, cross-validation',
            'Metrics (accuracy, precision, recall, RMSE), confusion matrices'
          ],
          asyncWork: [
            'Evaluate models from previous weeks',
            'Apply cross-validation',
            'Write up findings and recommendations'
          ],
          resources: [
            { title: 'Scikit-learn Model Evaluation', url: 'https://scikit-learn.org/stable/modules/model_evaluation.html', type: 'docs' },
            { title: 'StatQuest - Cross Validation', url: 'https://www.youtube.com/watch?v=fSytzGwwBVw', type: 'video' },
            { title: 'Google ML - Metrics', url: 'https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall', type: 'tutorial' }
          ],
          exercise: {
            id: 'ex-week6',
            title: 'Model Evaluation Deep Dive',
            description: 'Comprehensive model evaluation with cross-validation',
            starterCode: `import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Generate imbalanced dataset
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2,
                          weights=[0.9, 0.1], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# TODO: Complete evaluation
# 1. Calculate confusion matrix
# 2. Calculate precision, recall, F1
# 3. Perform 5-fold cross-validation
`,
            solution: `import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

X, y = make_classification(n_samples=1000, n_features=10, n_classes=2,
                          weights=[0.9, 0.1], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\\n", cm)

# Metrics
print(f"\\nAccuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(f"\\n5-Fold CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")`
          }
        },
        {
          id: 'week-7',
          title: 'Week 7: Unsupervised Learning & Recommendations',
          isRevised: true,
          sessions: [
            'Clustering (k-means), dimensionality reduction (PCA), similarity metrics',
            'Recommendation systems: collaborative filtering, content-based, matrix factorization'
          ],
          asyncWork: [
            'Build a movie/product recommendation system',
            'Practice preprocessing pipeline',
            'Handle missing data and normalization'
          ],
          resources: [
            { title: 'Scikit-learn Clustering', url: 'https://scikit-learn.org/stable/modules/clustering.html', type: 'docs' },
            { title: 'Google - Recommendation Systems', url: 'https://developers.google.com/machine-learning/recommendation', type: 'course' },
            { title: 'Surprise Library', url: 'https://surpriselib.com/', type: 'docs' }
          ],
          exercise: {
            id: 'ex-week7',
            title: 'Build a Recommendation System',
            description: 'Create a collaborative filtering recommender',
            starterCode: `import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# User-Item ratings matrix (0 = not rated)
ratings = np.array([
    [5, 3, 0, 1, 4],
    [4, 0, 0, 1, 5],
    [1, 1, 0, 5, 4],
    [0, 0, 5, 4, 0],
    [0, 4, 4, 0, 0],
])
movies = ['Action Hero', 'Comedy Night', 'Drama Queen', 'Horror House', 'Sci-Fi Space']

# TODO: Build collaborative filtering recommender
# 1. Calculate user-user similarity
# 2. Predict rating for User 3 on Movie 0
# 3. Generate recommendations for User 3
`,
            solution: `import numpy as np

ratings = np.array([
    [5, 3, 0, 1, 4],
    [4, 0, 0, 1, 5],
    [1, 1, 0, 5, 4],
    [0, 0, 5, 4, 0],
    [0, 4, 4, 0, 0],
])
movies = ['Action Hero', 'Comedy Night', 'Drama Queen', 'Horror House', 'Sci-Fi Space']

# User similarity using correlation
ratings_masked = np.where(ratings == 0, np.nan, ratings)

def calc_similarity(r1, r2):
    mask = ~np.isnan(r1) & ~np.isnan(r2)
    if mask.sum() < 2: return 0
    return np.corrcoef(r1[mask], r2[mask])[0, 1]

n_users = ratings.shape[0]
user_sim = np.zeros((n_users, n_users))
for i in range(n_users):
    for j in range(n_users):
        if i != j:
            user_sim[i, j] = calc_similarity(ratings_masked[i], ratings_masked[j])

print("User Similarity Matrix:")
print(np.round(user_sim, 2))

# Predict for User 3, Movie 0
target_user, target_movie = 3, 0
raters = ratings[:, target_movie] > 0
sims = user_sim[target_user, raters]
pred = np.dot(sims, ratings[raters, target_movie]) / np.abs(sims).sum()
print(f"\\nPredicted rating for User 3 on '{movies[0]}': {pred:.2f}")`
          }
        }
      ]
    },
    {
      id: 'module-3',
      title: 'Module 3: Deep Learning & Neural Networks',
      weeks: [
        {
          id: 'week-8',
          title: 'Week 8: Neural Network Fundamentals',
          hasCapstone: true,
          sessions: [
            'What is a neural network? Layers, weights, activation functions',
            'Training networks—loss functions, gradient descent, backpropagation'
          ],
          asyncWork: [
            'Experiment with simple neural network in Keras/TensorFlow',
            '★ Capstone Project: Receive guidelines and begin planning'
          ],
          resources: [
            { title: 'TensorFlow Tutorials', url: 'https://www.tensorflow.org/tutorials', type: 'tutorial' },
            { title: 'Keras Documentation', url: 'https://keras.io/guides/', type: 'docs' },
            { title: '3Blue1Brown - Neural Networks', url: 'https://www.youtube.com/watch?v=aircAruvnKk', type: 'video' }
          ],
          exercise: {
            id: 'ex-week8',
            title: 'Build Your First Neural Network',
            description: 'Create a neural network classifier with Keras',
            starterCode: `# Neural Network Concepts
# In practice, use TensorFlow/Keras

print("Neural Network Architecture:")
print("  Input: 2 features")
print("  Hidden 1: 16 neurons, ReLU")
print("  Hidden 2: 8 neurons, ReLU")
print("  Output: 1 neuron, Sigmoid")

# TODO: Explain why non-linear activations matter
`,
            solution: `print("Neural Network Architecture:")
print("  Input: 2 features")
print("  Hidden 1: 16 neurons, ReLU activation")
print("  Hidden 2: 8 neurons, ReLU activation")
print("  Output: 1 neuron, Sigmoid (binary classification)")

print("\\nWhy Non-Linear Activations?")
print("- Linear layers stacked are still linear")
print("- Non-linear activations enable learning complex patterns")
print("- ReLU: max(0, x) - simple, effective, avoids vanishing gradients")`
          }
        },
        {
          id: 'week-9',
          title: 'Week 9: Deep Learning Applications',
          sessions: [
            'Convolutional neural networks for images, transfer learning',
            'Sequence models, attention mechanisms, intro to transformers'
          ],
          asyncWork: [
            'Image classification mini-project using transfer learning',
            'Readings on transformer architecture',
            'Continue capstone project'
          ],
          resources: [
            { title: 'TensorFlow - Transfer Learning', url: 'https://www.tensorflow.org/tutorials/images/transfer_learning', type: 'tutorial' },
            { title: 'Illustrated Transformer', url: 'https://jalammar.github.io/illustrated-transformer/', type: 'article' },
            { title: 'Hugging Face Course', url: 'https://huggingface.co/learn/nlp-course', type: 'course' }
          ],
          exercise: {
            id: 'ex-week9',
            title: 'Transfer Learning Concepts',
            description: 'Understand transfer learning principles',
            starterCode: `# Transfer Learning Exercise
# Explain the concepts

# TODO: Answer these questions:
# 1. Why freeze base model layers initially?
# 2. When should you unfreeze layers?
# 3. Why does transfer learning work?
`,
            solution: `print("Transfer Learning Best Practices:")

print("\\n1. Why freeze base model layers?")
print("   - Pre-trained weights capture general features")
print("   - Prevents destroying learned representations")
print("   - Faster training, less data needed")

print("\\n2. When to unfreeze layers?")
print("   - Fine-tuning after initial training")
print("   - When you have sufficient data")
print("   - Typically unfreeze top layers only")

print("\\n3. Why does transfer learning work?")
print("   - Lower layers learn universal features")
print("   - Higher layers learn task-specific features")
print("   - Your task benefits from general understanding")`
          }
        }
      ]
    },
    {
      id: 'module-4',
      title: 'Module 4: Generative AI & LLMs',
      weeks: [
        {
          id: 'week-10',
          title: 'Week 10: Large Language Models',
          sessions: [
            'How LLMs work—tokenization, attention, training at scale',
            'Capabilities and limitations, hallucinations, reasoning patterns'
          ],
          asyncWork: [
            'Explore different LLMs',
            'Document observed behaviors and limitations',
            'Continue capstone project'
          ],
          resources: [
            { title: 'Anthropic - Claude Documentation', url: 'https://docs.anthropic.com/', type: 'docs' },
            { title: 'OpenAI - GPT Best Practices', url: 'https://platform.openai.com/docs/guides/prompt-engineering', type: 'docs' },
            { title: 'Attention Is All You Need', url: 'https://arxiv.org/abs/1706.03762', type: 'paper' }
          ],
          exercise: {
            id: 'ex-week10',
            title: 'LLM Behavior Analysis',
            description: 'Explore LLM capabilities and limitations',
            starterCode: `# LLM Analysis
# Test prompts with different LLMs

test_prompts = {
    "factual": "What is the capital of France?",
    "reasoning": "If all roses are flowers...",
    "math": "What is 17 * 24?",
}

# TODO: Document findings about:
# - Which tasks do LLMs excel at?
# - Where do they struggle?
`,
            solution: `findings = {
    "strengths": [
        "Factual recall for well-known info",
        "Creative writing and brainstorming",
        "Code generation and explanation",
        "Summarization and translation"
    ],
    "limitations": [
        "Math beyond simple arithmetic",
        "Recent events (knowledge cutoff)",
        "Consistent counting/listing",
        "Admitting uncertainty"
    ]
}

print("LLM Analysis:")
print("\\nStrengths:", findings["strengths"])
print("\\nLimitations:", findings["limitations"])`
          }
        },
        {
          id: 'week-11',
          title: 'Week 11: Applied GenAI & RAG',
          isRevised: true,
          sessions: [
            'Prompt engineering—chain-of-thought, structured outputs, few-shot learning',
            'RAG architecture, vector databases, embeddings; Intro to Agentic AI'
          ],
          asyncWork: [
            'Build a RAG-powered application using LangChain',
            'Implement document Q&A system',
            'Continue capstone project'
          ],
          resources: [
            { title: 'LangChain Documentation', url: 'https://python.langchain.com/docs/', type: 'docs' },
            { title: 'Pinecone - Vector DB Guide', url: 'https://www.pinecone.io/learn/', type: 'tutorial' },
            { title: 'ChromaDB', url: 'https://docs.trychroma.com/', type: 'docs' }
          ],
          exercise: {
            id: 'ex-week11',
            title: 'RAG Pipeline Concepts',
            description: 'Understand RAG architecture',
            starterCode: `# RAG Pipeline
# Understand each step

rag_steps = [
    "1. Load documents",
    "2. Chunk into smaller pieces",
    "3. Create embeddings",
    "4. Store in vector DB",
    "5. User query",
    "6. Retrieve relevant chunks",
    "7. Augment prompt with context",
    "8. Generate answer"
]

# TODO: Explain chunking strategy and retrieval
`,
            solution: `print("RAG Pipeline:")
print("1. Load documents (PDF, text, web)")
print("2. Chunk (500-1000 tokens, 10-20% overlap)")
print("3. Embed with model (OpenAI, sentence-transformers)")
print("4. Store in vector DB (Pinecone, Chroma, FAISS)")
print("5. User asks question")
print("6. Retrieve top-k similar chunks")
print("7. Add context to LLM prompt")
print("8. Generate grounded answer")

print("\\nBest Practices:")
print("- Chunk size: 500-1000 tokens")
print("- Overlap: 10-20% for context")
print("- Retrieve k=3-5 chunks")
print("- Re-rank for relevance")`
          }
        }
      ]
    },
    {
      id: 'module-5',
      title: 'Module 5: Capstone & Career',
      weeks: [
        {
          id: 'week-12',
          title: 'Week 12: Capstone Presentations',
          sessions: [
            'Student capstone presentations (first half)',
            'Student capstone presentations (second half), peer feedback'
          ],
          asyncWork: [
            'Finalize capstone project',
            'Prepare 10-minute presentation',
            'Portfolio and resume review',
            'Career paths and continued learning'
          ],
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
