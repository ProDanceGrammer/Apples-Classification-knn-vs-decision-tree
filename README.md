# Apples Quality Classification — KNN vs Decision Tree

This project compares two classical machine learning algorithms — **K-Nearest Neighbors (KNN)** and **Decision Tree Classifier** — for predicting the quality of apples based on a dataset of physical and chemical characteristics.

The goal is to evaluate which model performs better and to provide a clean, modular project structure suitable for learning and portfolio use.

---

### Features

- Fully reproducible pipeline (data → preprocessing → modeling → evaluation)
- Visualization modules (histograms, boxplots, scatter plots)
- Comparison of KNN and Decision Tree models
- Modular and clean project structure
- YAML-based configuration system
- Easy to extend with new models or datasets


## Project Structure
```
Apples-Classification-knn-vs-decision-tree/
│
├── data/
│ └── apple_quality.csv
│
├── src/
│ ├── data/
│ │ └── load_data.py
│ │ └── preprocess.py
│ │ └── split.py
│ ├── models/
│ │ └── decision_tree_model.py
│ │ └── knn_model.py
│ │ └── metrics.py
│ ├── visualization/
│ │ └── boxplots.py
│ │ └── histograms.py
│ │ └── scatter_plots.py
│ └── config_loader.py
│ └── main.py
│
├── configs/
│ └── config.yaml
│
├── notebooks/
│ └── Apples_classification.ipynb
│
├── report/
│ └── Presentation report.pdf
│ └── Technical report.pdf
│
├── requirements.txt
└── README.md
```


---

##  How to Run the Project

### 1. Clone the repository
```
git clone https://github.com/Prodancegrammer/Apples-Classification-knn-vs-decision-tree.git
cd Apples-Classification-knn-vs-decision-tree
```

### 2. Create a virtual environment
```
python3 -m venv venv
source venv/bin/activate     # macOS / Linux
venv\Scripts\activate        # Windows
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Run the main script
```
python src/main.py
```


### Configuration (config.yaml)
All project settings — model parameters, and visualization options — are stored in:
```
configs/config.yaml
```

Example:
```
seed: 210029807
test_size: 0.2

model:
  n_neighbors: 9
  max_depth: 9
  min_samples_split: 2
```


### Visualizations

The folder:
```
src/visualization
```
contains all plotting modules with functions, including:

- Histograms of numerical features
- Boxplots for detecting outliers
- Scatter plots to find the correlations

Plots are generated using Matplotlib and Seaborn.



### Machine Learning Models Used

This project uses:

- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- StandardScaler
- Train/Test Split
- Accuracy, Precision, Recall, F1-score

The task is binary classification — predicting whether an apple is "good" or "bad".


### Tech Stack

- Python 3.10+
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- PyYAML



## Author

Vasyl Yarmolenko

Machine Learning Engineer / Data Science enthusiast

GitHub: https://github.com/Prodancegrammer

LinkedIn: https://www.linkedin.com/in/vasyl-yarmolenko-ba427a255/
