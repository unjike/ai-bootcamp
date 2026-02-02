// AI Fundamentals Bootcamp - Curriculum Data
// Version: 3.1 - Removed exercises, expanded quizzes to 10 questions

export const curriculum = {
  preWork: {
    id: 'pre-work',
    title: 'Pre-Work: AI Ethics & Responsible AI',
    duration: '~8 hours (self-paced)',
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
    ]
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
            'Reflection exercise on AI in their own field',
            'Connect learnings to pre-work ethics module'
          ],
          resources: [
            { title: 'Elements of AI - Free Course', url: 'https://www.elementsofai.com/', type: 'course' },
            { title: 'Google AI Crash Course', url: 'https://developers.google.com/machine-learning/crash-course', type: 'course' },
            { title: 'AI Timeline - Our World in Data', url: 'https://ourworldindata.org/brief-history-of-ai', type: 'article' }
          ]
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
            'Practice data cleaning techniques',
            'Create visualizations to communicate findings'
          ],
          project: {
            title: 'EDA on IBM HR Attrition Dataset',
            description: 'Data cleaning, visualization, business recommendations'
          },
          resources: [
            { title: 'Pandas Documentation', url: 'https://pandas.pydata.org/docs/', type: 'docs' },
            { title: 'Kaggle - Data Cleaning', url: 'https://www.kaggle.com/learn/data-cleaning', type: 'tutorial' },
            { title: 'Matplotlib Tutorial', url: 'https://matplotlib.org/stable/tutorials/index.html', type: 'tutorial' },
            { title: 'IBM HR Attrition Dataset', url: 'https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset', type: 'tool' }
          ]
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
          project: {
            title: 'TechMart SQL Data Pipeline',
            description: '\$2.4M transaction analysis, customer segmentation, cohort analysis'
          },
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
          ]
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
          project: {
            title: 'Lending Club Interest Rate Prediction (Part A)',
            description: 'Predict interest rates on 890K+ loans using regression techniques'
          },
          resources: [
            { title: 'Scikit-learn Linear Models', url: 'https://scikit-learn.org/stable/modules/linear_model.html', type: 'docs' },
            { title: 'StatQuest - Linear Regression', url: 'https://www.youtube.com/watch?v=nk2CQITm_eo', type: 'video' },
            { title: 'Lending Club Dataset', url: 'https://www.kaggle.com/datasets/wordsforthewise/lending-club', type: 'tool' }
          ]
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
          project: {
            title: 'Lending Club Default Prediction (Part B)',
            description: 'Build classifier with AUC 0.75+ on imbalanced data. Each prevented default saves ~\$9,000'
          },
          resources: [
            { title: 'Scikit-learn Classification', url: 'https://scikit-learn.org/stable/modules/tree.html', type: 'docs' },
            { title: 'StatQuest - Decision Trees', url: 'https://www.youtube.com/watch?v=_L39rN6gz7Y', type: 'video' },
            { title: 'StatQuest - Random Forests', url: 'https://www.youtube.com/watch?v=J4Wdy0Wc_xQ', type: 'video' }
          ]
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
          project: {
            title: 'Model Evaluation & Streamlit Deployment (Parts C & D)',
            description: 'Complete model evaluation with cross-validation. Part D (Bonus): Deploy as interactive Streamlit web application'
          },
          optionalModule: {
            title: 'Time Series Forecasting',
            description: 'Introduction to time series data, trend and seasonality, ARIMA basics, forecasting with Prophet. For finance, supply chain, or demand forecasting roles.'
          },
          resources: [
            { title: 'Scikit-learn Model Evaluation', url: 'https://scikit-learn.org/stable/modules/model_evaluation.html', type: 'docs' },
            { title: 'StatQuest - Cross Validation', url: 'https://www.youtube.com/watch?v=fSytzGwwBVw', type: 'video' },
            { title: 'Streamlit Documentation', url: 'https://docs.streamlit.io/', type: 'docs' }
          ]
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
          project: {
            title: 'MovieLens Recommendation System',
            description: 'Hybrid recommender combining collaborative filtering and content-based approaches on 100K ratings'
          },
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
          ]
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
          project: {
            title: 'Heart Disease Prediction with Neural Networks',
            description: 'UCI dataset, compare NN vs traditional ML, \$75K per missed diagnosis business impact'
          },
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
          ]
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
          project: {
            title: 'Image Classification with Transfer Learning',
            description: 'Kaggle dataset of student choice, demonstrate 80%+ reduction in training time'
          },
          keyBridge: 'Week 9 connects traditional deep learning to modern LLMs by explaining why Transformers replaced RNNs (parallelization, direct long-range connections)',
          resources: [
            { title: 'TensorFlow Transfer Learning', url: 'https://www.tensorflow.org/tutorials/images/transfer_learning', type: 'tutorial' },
            { title: 'Illustrated Transformer', url: 'https://jalammar.github.io/illustrated-transformer/', type: 'article' },
            { title: 'Kaggle Image Classification', url: 'https://www.kaggle.com/competitions?search=image+classification', type: 'tool' }
          ]
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
          project: {
            title: 'LLM-Powered Customer Support System',
            description: 'Ticket classification (85%+ accuracy) and response generation using prompt engineering, \$500K+ annual savings potential'
          },
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
          ]
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
          project: {
            title: 'Document Q&A System',
            description: 'RAG-powered company knowledge base that answers questions about uploaded documents'
          },
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
          ]
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
          ]
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

// Quizzes - 10 questions each, 70% to pass (7/10)
export const quizzes = {
  'pre-work': {
    id: 'quiz-prework',
    title: 'AI Ethics Quiz',
    passingScore: 70,
    questions: [
      { id: 'q1', question: 'What is algorithmic bias?', options: ['A type of computer virus', 'Systematic errors in AI systems that create unfair outcomes', 'A programming language', 'A hardware malfunction'], correctAnswer: 1 },
      { id: 'q2', question: 'Which is a key principle of responsible AI?', options: ['Maximizing profit at all costs', 'Hiding how AI systems make decisions', 'Transparency and explainability', 'Using data without consent'], correctAnswer: 2 },
      { id: 'q3', question: 'What is differential privacy?', options: ['A technique to protect individual data while allowing useful analysis', 'A way to make AI faster', 'A type of neural network', 'A programming framework'], correctAnswer: 0 },
      { id: 'q4', question: 'Why is AI fairness important?', options: ['It makes code run faster', 'It ensures AI systems do not discriminate against protected groups', 'It reduces server costs', 'It is not actually important'], correctAnswer: 1 },
      { id: 'q5', question: 'What should you do if you discover bias in your AI model?', options: ['Ignore it and deploy anyway', 'Investigate the source, mitigate, and document', 'Delete all the data', 'Blame the data provider'], correctAnswer: 1 },
      { id: 'q6', question: 'What was the issue with Amazons AI hiring tool?', options: ['It was too slow', 'It discriminated against women candidates', 'It cost too much', 'It only worked in English'], correctAnswer: 1 },
      { id: 'q7', question: 'What is explainability in AI?', options: ['Writing code comments', 'The ability to understand how an AI reached its decision', 'Making AI run faster', 'A type of testing'], correctAnswer: 1 },
      { id: 'q8', question: 'Which group should be involved in AI ethics decisions?', options: ['Only engineers', 'Only executives', 'Diverse stakeholders including affected communities', 'Only the legal team'], correctAnswer: 2 },
      { id: 'q9', question: 'What is data anonymization?', options: ['Deleting all data', 'Removing personally identifiable information', 'Encrypting databases', 'Backing up data'], correctAnswer: 1 },
      { id: 'q10', question: 'What is the COMPAS case study about?', options: ['Image recognition bias', 'Racial bias in criminal risk assessment AI', 'Voice assistant privacy', 'Self-driving car safety'], correctAnswer: 1 }
    ]
  },
  'week-1': {
    id: 'quiz-week1',
    title: 'What Is AI? Quiz',
    passingScore: 70,
    questions: [
      { id: 'q1', question: 'What is Machine Learning?', options: ['A type of robot', 'A subset of AI where systems learn from data without explicit programming', 'A programming language', 'A database system'], correctAnswer: 1 },
      { id: 'q2', question: 'Which is NOT a type of machine learning?', options: ['Supervised learning', 'Unsupervised learning', 'Reinforcement learning', 'Mechanical learning'], correctAnswer: 3 },
      { id: 'q3', question: 'What is Deep Learning?', options: ['Learning while sleeping', 'Machine learning using neural networks with many layers', 'A database query technique', 'A hardware component'], correctAnswer: 1 },
      { id: 'q4', question: 'What does training mean in ML?', options: ['Teaching humans to code', 'A model learning patterns from data', 'Installing software updates', 'Writing documentation'], correctAnswer: 1 },
      { id: 'q5', question: 'Which company created ChatGPT?', options: ['Google', 'Meta', 'OpenAI', 'Microsoft'], correctAnswer: 2 },
      { id: 'q6', question: 'What is supervised learning?', options: ['Learning with a human watching', 'Learning from labeled data with known outputs', 'Learning without any data', 'Learning from rewards'], correctAnswer: 1 },
      { id: 'q7', question: 'What is the difference between AI and ML?', options: ['They are the same thing', 'ML is a subset of AI focused on learning from data', 'AI is a subset of ML', 'They are unrelated'], correctAnswer: 1 },
      { id: 'q8', question: 'What is generative AI?', options: ['AI that generates new content like text, images, or code', 'AI that only classifies data', 'AI used for gaming', 'AI for spreadsheets'], correctAnswer: 0 },
      { id: 'q9', question: 'What year did the term Artificial Intelligence originate?', options: ['1936', '1956', '1976', '1996'], correctAnswer: 1 },
      { id: 'q10', question: 'What is unsupervised learning used for?', options: ['Classification with labels', 'Finding patterns in unlabeled data', 'Playing games', 'Web development'], correctAnswer: 1 }
    ]
  },
  'week-2': {
    id: 'quiz-week2',
    title: 'Data Thinking Quiz',
    passingScore: 70,
    questions: [
      { id: 'q1', question: 'What is EDA?', options: ['A programming language', 'Exploratory Data Analysis - analyzing datasets to summarize characteristics', 'A database system', 'A machine learning model'], correctAnswer: 1 },
      { id: 'q2', question: 'Which is NOT a data quality issue?', options: ['Missing values', 'Duplicate records', 'Data being too accurate', 'Inconsistent formatting'], correctAnswer: 2 },
      { id: 'q3', question: 'What does the median represent?', options: ['The average value', 'The most frequent value', 'The middle value when data is sorted', 'The range of values'], correctAnswer: 2 },
      { id: 'q4', question: 'Why is data visualization important?', options: ['Makes reports longer', 'Helps identify patterns and communicate insights', 'Slows down analysis', 'Not actually important'], correctAnswer: 1 },
      { id: 'q5', question: 'What dataset is used in the Week 2 project?', options: ['MovieLens', 'Lending Club', 'IBM HR Attrition', 'MNIST'], correctAnswer: 2 },
      { id: 'q6', question: 'What is the difference between mean and median?', options: ['They are the same', 'Mean is the average, median is the middle value', 'Median is the average, mean is the middle', 'Neither measures central tendency'], correctAnswer: 1 },
      { id: 'q7', question: 'What is an outlier?', options: ['A common data point', 'A data point significantly different from others', 'A missing value', 'A duplicate record'], correctAnswer: 1 },
      { id: 'q8', question: 'Which chart is best for showing distribution of a single variable?', options: ['Scatter plot', 'Histogram', 'Line chart', 'Pie chart'], correctAnswer: 1 },
      { id: 'q9', question: 'What is data cleaning?', options: ['Deleting all data', 'Fixing errors, handling missing values, removing duplicates', 'Making visualizations', 'Training models'], correctAnswer: 1 },
      { id: 'q10', question: 'What does correlation measure?', options: ['Causation between variables', 'The strength of relationship between variables', 'Data quality', 'Processing speed'], correctAnswer: 1 }
    ]
  },
  'week-3': {
    id: 'quiz-week3',
    title: 'Python & SQL Quiz',
    passingScore: 70,
    questions: [
      { id: 'q1', question: 'What does SQL stand for?', options: ['Simple Question Language', 'Structured Query Language', 'System Quality Level', 'Standard Query Logic'], correctAnswer: 1 },
      { id: 'q2', question: 'Which SQL clause filters results?', options: ['SELECT', 'FROM', 'WHERE', 'ORDER BY'], correctAnswer: 2 },
      { id: 'q3', question: 'What does a JOIN do in SQL?', options: ['Deletes tables', 'Combines rows from two or more tables', 'Creates a new database', 'Sorts data alphabetically'], correctAnswer: 1 },
      { id: 'q4', question: 'Which Python data structure uses key-value pairs?', options: ['List', 'Tuple', 'Dictionary', 'Set'], correctAnswer: 2 },
      { id: 'q5', question: 'What is the TechMart project transaction value?', options: ['$500K', '$1.2M', '$2.4M', '$5M'], correctAnswer: 2 },
      { id: 'q6', question: 'What does GROUP BY do in SQL?', options: ['Sorts results', 'Filters rows', 'Aggregates data by categories', 'Joins tables'], correctAnswer: 2 },
      { id: 'q7', question: 'What is a CTE in SQL?', options: ['Common Table Expression - a temporary named result set', 'Column Type Extension', 'Central Table Entity', 'Custom Text Encoding'], correctAnswer: 0 },
      { id: 'q8', question: 'Which Python library is used for data manipulation?', options: ['NumPy', 'Pandas', 'Matplotlib', 'Scikit-learn'], correctAnswer: 1 },
      { id: 'q9', question: 'What does SELECT * mean?', options: ['Select nothing', 'Select all columns', 'Select the first row', 'Delete all data'], correctAnswer: 1 },
      { id: 'q10', question: 'What is a window function in SQL?', options: ['A function that opens windows', 'A function that performs calculations across rows related to current row', 'A function for creating views', 'A function for deleting data'], correctAnswer: 1 }
    ]
  },
  'week-4': {
    id: 'quiz-week4',
    title: 'Regression Quiz',
    passingScore: 70,
    questions: [
      { id: 'q1', question: 'What is linear regression used for?', options: ['Classification tasks', 'Predicting continuous numerical values', 'Clustering data', 'Image recognition'], correctAnswer: 1 },
      { id: 'q2', question: 'What is a feature in machine learning?', options: ['The output we want to predict', 'An input variable used for prediction', 'A type of algorithm', 'A visualization technique'], correctAnswer: 1 },
      { id: 'q3', question: 'Which metric is commonly used to evaluate regression models?', options: ['Accuracy', 'RMSE (Root Mean Squared Error)', 'F1 Score', 'Precision'], correctAnswer: 1 },
      { id: 'q4', question: 'What does the Lending Club Part A project predict?', options: ['Loan defaults', 'Interest rates', 'Customer satisfaction', 'Stock prices'], correctAnswer: 1 },
      { id: 'q5', question: 'What does a negative coefficient mean in linear regression?', options: ['The model failed', 'As the feature increases, the target decreases', 'The feature is not important', 'There is an error in the data'], correctAnswer: 1 },
      { id: 'q6', question: 'What is R-squared?', options: ['A programming language', 'A measure of how well the model explains variance in the target', 'A type of neural network', 'A data cleaning technique'], correctAnswer: 1 },
      { id: 'q7', question: 'What is feature engineering?', options: ['Building computer hardware', 'Creating or transforming features to improve model performance', 'A type of testing', 'Database design'], correctAnswer: 1 },
      { id: 'q8', question: 'What is the target variable?', options: ['The input features', 'The variable we want to predict', 'The training data', 'The test data'], correctAnswer: 1 },
      { id: 'q9', question: 'What is multicollinearity?', options: ['Multiple models', 'High correlation between independent variables', 'A type of neural network', 'A visualization'], correctAnswer: 1 },
      { id: 'q10', question: 'What library provides LinearRegression in Python?', options: ['Pandas', 'NumPy', 'Scikit-learn', 'TensorFlow'], correctAnswer: 2 }
    ]
  },
  'week-5': {
    id: 'quiz-week5',
    title: 'Classification Quiz',
    passingScore: 70,
    questions: [
      { id: 'q1', question: 'What is classification used for?', options: ['Predicting continuous values', 'Predicting categorical outcomes', 'Grouping unlabeled data', 'Reducing dimensions'], correctAnswer: 1 },
      { id: 'q2', question: 'What is a Random Forest?', options: ['A single decision tree', 'An ensemble of many decision trees', 'A type of data', 'A visualization tool'], correctAnswer: 1 },
      { id: 'q3', question: 'What does AUC-ROC measure?', options: ['Processing speed', 'Classification performance across all thresholds', 'Data size', 'Feature importance only'], correctAnswer: 1 },
      { id: 'q4', question: 'How much does each prevented default save in the project?', options: ['$1,000', '$5,000', '$9,000', '$50,000'], correctAnswer: 2 },
      { id: 'q5', question: 'What is class imbalance?', options: ['Different feature scales', 'One class being much more frequent than others', 'Complex model architecture', 'Missing data in classes'], correctAnswer: 1 },
      { id: 'q6', question: 'What is logistic regression used for?', options: ['Regression problems', 'Binary classification problems', 'Clustering', 'Dimensionality reduction'], correctAnswer: 1 },
      { id: 'q7', question: 'What is precision in classification?', options: ['The proportion of true positives among predicted positives', 'The proportion of true positives among actual positives', 'Overall accuracy', 'Processing time'], correctAnswer: 0 },
      { id: 'q8', question: 'What is recall in classification?', options: ['The proportion of true positives among predicted positives', 'The proportion of true positives among actual positives', 'Overall accuracy', 'Memory usage'], correctAnswer: 1 },
      { id: 'q9', question: 'What technique helps with class imbalance?', options: ['Removing all minority class samples', 'SMOTE (Synthetic Minority Oversampling)', 'Using only accuracy metric', 'Ignoring the problem'], correctAnswer: 1 },
      { id: 'q10', question: 'What is a decision boundary?', options: ['The edge of a dataset', 'The line or surface separating different classes', 'A type of error', 'A preprocessing step'], correctAnswer: 1 }
    ]
  },
  'week-6': {
    id: 'quiz-week6',
    title: 'Model Evaluation & Deployment Quiz',
    passingScore: 70,
    questions: [
      { id: 'q1', question: 'What is overfitting?', options: ['Good performance on new data', 'Model learns noise, performs poorly on new data', 'Model is too simple', 'Training takes too long'], correctAnswer: 1 },
      { id: 'q2', question: 'What is cross-validation used for?', options: ['Cleaning data', 'Evaluating model performance more robustly', 'Creating visualizations', 'Faster training'], correctAnswer: 1 },
      { id: 'q3', question: 'When should you prioritize recall over precision?', options: ['When false positives are very costly', 'When false negatives are very costly', 'Always', 'Never'], correctAnswer: 1 },
      { id: 'q4', question: 'What is Streamlit used for?', options: ['Training ML models', 'Building interactive web applications', 'Data cleaning', 'Database management'], correctAnswer: 1 },
      { id: 'q5', question: 'What does a confusion matrix show?', options: ['Model parameters', 'Predictions vs actual values (TP, TN, FP, FN)', 'Training time', 'Feature correlations'], correctAnswer: 1 },
      { id: 'q6', question: 'What is k-fold cross-validation?', options: ['Training k different models', 'Splitting data into k parts and rotating the test set', 'Using k features', 'Running for k iterations'], correctAnswer: 1 },
      { id: 'q7', question: 'What is hyperparameter tuning?', options: ['Adjusting model weights', 'Finding optimal configuration settings for the model', 'Cleaning the data', 'Feature engineering'], correctAnswer: 1 },
      { id: 'q8', question: 'What is model serialization?', options: ['Training in parallel', 'Saving a trained model to disk for later use', 'Data preprocessing', 'Feature selection'], correctAnswer: 1 },
      { id: 'q9', question: 'What is the train-test split for?', options: ['Making training faster', 'Evaluating model on unseen data', 'Reducing data size', 'Data visualization'], correctAnswer: 1 },
      { id: 'q10', question: 'What is underfitting?', options: ['Model is too complex', 'Model is too simple to capture patterns', 'Training takes too long', 'Perfect model performance'], correctAnswer: 1 }
    ]
  },
  'week-7': {
    id: 'quiz-week7',
    title: 'Unsupervised Learning & Recommendations Quiz',
    passingScore: 70,
    questions: [
      { id: 'q1', question: 'What is clustering?', options: ['Predicting labels', 'Grouping similar data points together', 'Training neural networks', 'Creating visualizations'], correctAnswer: 1 },
      { id: 'q2', question: 'What is collaborative filtering?', options: ['Filtering spam emails', 'Recommending items based on similar users or items', 'Cleaning data', 'A clustering method'], correctAnswer: 1 },
      { id: 'q3', question: 'What is the cold start problem?', options: ['Servers being too cold', 'Difficulty recommending for new users or items with no history', 'Slow computation', 'Storage limitations'], correctAnswer: 1 },
      { id: 'q4', question: 'What dataset is used in the Week 7 project?', options: ['Lending Club', 'MovieLens 100K', 'IBM HR Attrition', 'MNIST'], correctAnswer: 1 },
      { id: 'q5', question: 'What does PCA stand for?', options: ['Primary Component Analysis', 'Principal Component Analysis', 'Partial Cluster Algorithm', 'Predictive Classification Approach'], correctAnswer: 1 },
      { id: 'q6', question: 'What is K-Means clustering?', options: ['A classification algorithm', 'An algorithm that partitions data into k clusters', 'A regression technique', 'A neural network type'], correctAnswer: 1 },
      { id: 'q7', question: 'What is cosine similarity used for?', options: ['Measuring angle-based similarity between vectors', 'Calculating distances', 'Training models', 'Data cleaning'], correctAnswer: 0 },
      { id: 'q8', question: 'What is content-based filtering?', options: ['Filtering by file type', 'Recommending based on item attributes and features', 'Removing content', 'A clustering method'], correctAnswer: 1 },
      { id: 'q9', question: 'What is matrix factorization (SVD) used for in recommendations?', options: ['Data cleaning', 'Decomposing user-item matrix to find latent factors', 'Visualization', 'Classification'], correctAnswer: 1 },
      { id: 'q10', question: 'What is the elbow method?', options: ['A physical exercise', 'A technique to find optimal number of clusters', 'A data cleaning method', 'A visualization type'], correctAnswer: 1 }
    ]
  },
  'week-8': {
    id: 'quiz-week8',
    title: 'Neural Networks Quiz',
    passingScore: 70,
    questions: [
      { id: 'q1', question: 'What is an activation function?', options: ['Starts the training process', 'Introduces non-linearity to neural networks', 'Loads the data', 'Calculates the loss'], correctAnswer: 1 },
      { id: 'q2', question: 'What is backpropagation?', options: ['Moving data backwards', 'Algorithm for calculating gradients to update weights', 'A type of neural network', 'Data preprocessing'], correctAnswer: 1 },
      { id: 'q3', question: 'What is the cost of a missed heart disease diagnosis in the project?', options: ['$5K', '$25K', '$75K', '$500K'], correctAnswer: 2 },
      { id: 'q4', question: 'When should you choose a neural network over logistic regression?', options: ['Small data, linear relationships', 'Large data, complex non-linear patterns', 'Always', 'Never'], correctAnswer: 1 },
      { id: 'q5', question: 'What does ReLU activation do?', options: ['Returns input if positive, otherwise 0', 'Squashes values to 0-1', 'Normalizes data', 'Calculates loss'], correctAnswer: 0 },
      { id: 'q6', question: 'What is a hidden layer?', options: ['A layer that is not visible in code', 'A layer between input and output layers', 'The output layer', 'The input layer'], correctAnswer: 1 },
      { id: 'q7', question: 'What is gradient descent?', options: ['A type of layer', 'An optimization algorithm to minimize loss', 'A data preprocessing step', 'A visualization technique'], correctAnswer: 1 },
      { id: 'q8', question: 'What is the purpose of a loss function?', options: ['To add more layers', 'To measure how wrong predictions are', 'To load data', 'To visualize results'], correctAnswer: 1 },
      { id: 'q9', question: 'What framework is commonly used for neural networks?', options: ['Pandas', 'Scikit-learn', 'Keras and TensorFlow', 'Matplotlib'], correctAnswer: 2 },
      { id: 'q10', question: 'What is an epoch?', options: ['A single data point', 'One complete pass through the training data', 'A type of layer', 'An activation function'], correctAnswer: 1 }
    ]
  },
  'week-9': {
    id: 'quiz-week9',
    title: 'Deep Learning Applications Quiz',
    passingScore: 70,
    questions: [
      { id: 'q1', question: 'What is a CNN primarily used for?', options: ['Text processing', 'Image and visual data processing', 'Time series only', 'Recommendations'], correctAnswer: 1 },
      { id: 'q2', question: 'What is transfer learning?', options: ['Moving data between servers', 'Using a pre-trained model for a new task', 'Data augmentation', 'File transfer'], correctAnswer: 1 },
      { id: 'q3', question: 'Why freeze base layers in transfer learning?', options: ['To train faster and preserve learned features', 'To increase model size', 'To add more data', 'For visualization'], correctAnswer: 0 },
      { id: 'q4', question: 'What did Transformers replace in sequence modeling?', options: ['Convolutions', 'RNNs and LSTMs with self-attention', 'Pooling layers', 'Dropout'], correctAnswer: 1 },
      { id: 'q5', question: 'How much training time reduction does transfer learning provide?', options: ['10-20%', '30-50%', '80% or more reduction', 'No reduction'], correctAnswer: 2 },
      { id: 'q6', question: 'What is a convolutional layer?', options: ['A layer that applies filters to detect features', 'A fully connected layer', 'An output layer', 'A dropout layer'], correctAnswer: 0 },
      { id: 'q7', question: 'What is data augmentation?', options: ['Deleting data', 'Creating variations of training data to increase dataset size', 'Data cleaning', 'Feature selection'], correctAnswer: 1 },
      { id: 'q8', question: 'What is VGG16?', options: ['A database', 'A pre-trained CNN model for image classification', 'A programming language', 'A loss function'], correctAnswer: 1 },
      { id: 'q9', question: 'What problem do LSTMs solve that basic RNNs struggle with?', options: ['Image processing', 'Long-range dependencies and vanishing gradient', 'Fast training', 'Small datasets'], correctAnswer: 1 },
      { id: 'q10', question: 'What is attention mechanism in deep learning?', options: ['User attention tracking', 'Allowing models to focus on relevant parts of input', 'A regularization technique', 'A loss function'], correctAnswer: 1 }
    ]
  },
  'week-10': {
    id: 'quiz-week10',
    title: 'LLMs & Prompt Engineering Quiz',
    passingScore: 70,
    questions: [
      { id: 'q1', question: 'What is tokenization?', options: ['Encrypting data', 'Breaking text into smaller units called tokens', 'Model training', 'Data visualization'], correctAnswer: 1 },
      { id: 'q2', question: 'What is an LLM hallucination?', options: ['A visual effect', 'When an LLM generates false or fabricated information', 'A training technique', 'A type of architecture'], correctAnswer: 1 },
      { id: 'q3', question: 'What is chain-of-thought prompting?', options: ['Linking multiple models', 'Asking the AI to reason step-by-step', 'A database technique', 'Data visualization'], correctAnswer: 1 },
      { id: 'q4', question: 'What is the potential annual savings in the LLM Customer Support project?', options: ['$50K', '$100K', '$500K or more', '$10M'], correctAnswer: 2 },
      { id: 'q5', question: 'Who created Claude?', options: ['OpenAI', 'Google', 'Anthropic', 'Meta'], correctAnswer: 2 },
      { id: 'q6', question: 'What is zero-shot prompting?', options: ['Training from scratch', 'Asking the model to perform a task without examples', 'Using zero data', 'A fine-tuning method'], correctAnswer: 1 },
      { id: 'q7', question: 'What is few-shot prompting?', options: ['Using very little data for training', 'Providing a few examples in the prompt', 'Running for few iterations', 'A compression technique'], correctAnswer: 1 },
      { id: 'q8', question: 'What does temperature control in LLMs?', options: ['Server cooling', 'Randomness and creativity of outputs', 'Training speed', 'Model size'], correctAnswer: 1 },
      { id: 'q9', question: 'What is a knowledge cutoff in LLMs?', options: ['Data deletion', 'The date after which the model has no training data', 'Model size limit', 'Token limit'], correctAnswer: 1 },
      { id: 'q10', question: 'What is the recommended prompt structure?', options: ['Just ask the question', 'ROLE then CONTEXT then TASK then FORMAT then CONSTRAINTS', 'Use only keywords', 'Write in code'], correctAnswer: 1 }
    ]
  },
  'week-11': {
    id: 'quiz-week11',
    title: 'RAG & Agentic AI Quiz',
    passingScore: 70,
    questions: [
      { id: 'q1', question: 'What does RAG stand for?', options: ['Random Access Generation', 'Retrieval Augmented Generation', 'Rapid AI Growth', 'Recursive Algorithm Generation'], correctAnswer: 1 },
      { id: 'q2', question: 'What is a vector database used for in RAG?', options: ['Storing images only', 'Storing and searching text embeddings efficiently', 'Running SQL queries', 'Training models'], correctAnswer: 1 },
      { id: 'q3', question: 'What is the typical chunk size for RAG documents?', options: ['50-100 tokens', '500-1000 tokens', '5000-10000 tokens', 'Entire documents'], correctAnswer: 1 },
      { id: 'q4', question: 'Which is NOT a vector database?', options: ['ChromaDB', 'FAISS', 'Pinecone', 'PostgreSQL'], correctAnswer: 3 },
      { id: 'q5', question: 'What is an AI agent?', options: ['A human using AI', 'AI that can take actions and use tools autonomously', 'A database system', 'A visualization tool'], correctAnswer: 1 },
      { id: 'q6', question: 'What are embeddings?', options: ['HTML elements', 'Dense vector representations of text or data', 'Database indexes', 'Image files'], correctAnswer: 1 },
      { id: 'q7', question: 'Why is RAG useful?', options: ['Makes models smaller', 'Allows LLMs to access external and updated knowledge', 'Speeds up training', 'Reduces cost only'], correctAnswer: 1 },
      { id: 'q8', question: 'What is LangChain?', options: ['A blockchain', 'A framework for building LLM applications', 'A database', 'A programming language'], correctAnswer: 1 },
      { id: 'q9', question: 'What is the RAG pipeline order?', options: ['Generate then Retrieve then Augment', 'Retrieve then Augment then Generate', 'Augment then Generate then Retrieve', 'Random order'], correctAnswer: 1 },
      { id: 'q10', question: 'What is function calling in Agentic AI?', options: ['Calling Python functions', 'LLMs invoking external tools and APIs to take actions', 'Phone calls', 'Debugging'], correctAnswer: 1 }
    ]
  },
  'week-12': {
    id: 'quiz-week12',
    title: 'Capstone & Career Quiz',
    passingScore: 70,
    questions: [
      { id: 'q1', question: 'What should a good data science portfolio include?', options: ['Only code files', 'Projects with documentation, business impact, and clean code', 'Only certifications', 'Personal photos'], correctAnswer: 1 },
      { id: 'q2', question: 'Why is GitHub important for data scientists?', options: ['Social media presence', 'Showcasing code, collaboration, and version control', 'Storing personal files', 'Watching videos'], correctAnswer: 1 },
      { id: 'q3', question: 'What is the typical Data Scientist salary range?', options: ['$30K-$50K', '$88K-$110K', '$200K-$300K', '$500K or more'], correctAnswer: 1 },
      { id: 'q4', question: 'What should you emphasize in data science interviews?', options: ['Memorized definitions only', 'Problem-solving approach and communication skills', 'Typing speed', 'Number of certifications'], correctAnswer: 1 },
      { id: 'q5', question: 'How many portfolio projects are in this bootcamp?', options: ['5', '8', '11', '15'], correctAnswer: 1 },
      { id: 'q6', question: 'What is the STAR method for interviews?', options: ['A rating system', 'Situation, Task, Action, Result for answering behavioral questions', 'A programming framework', 'A data visualization'], correctAnswer: 1 },
      { id: 'q7', question: 'What should a good README include?', options: ['Just the code', 'Project overview, setup instructions, results, and documentation', 'Only images', 'Personal information'], correctAnswer: 1 },
      { id: 'q8', question: 'How long should the capstone presentation be?', options: ['5 minutes', '10 minutes', '30 minutes', '1 hour'], correctAnswer: 1 },
      { id: 'q9', question: 'What makes a capstone project stand out?', options: ['Using the most complex algorithm', 'Clear business impact and practical application', 'Length of code', 'Number of libraries used'], correctAnswer: 1 },
      { id: 'q10', question: 'Why is continuous learning important in AI and ML?', options: ['It is not important', 'The field evolves rapidly with new techniques and tools', 'To get more certifications', 'To impress recruiters'], correctAnswer: 1 }
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

// Simplified unlock logic - only requires quiz completion (no exercises)
export const isWeekUnlocked = (weekId, completedQuizzes) => {
  const weekIndex = getWeekIndex(weekId);
  if (weekIndex <= 1) return true; // Pre-work and Week 1 always unlocked
  
  // Check all previous weeks have passed quizzes
  for (let i = 0; i < weekIndex; i++) {
    const prevWeekId = weekUnlockOrder[i];
    if (!completedQuizzes[prevWeekId]) return false;
  }
  return true;
};
