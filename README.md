# BODY-PERFORMANCE-ANALYSIS

Analyze and predict human body performance using Python, data science, and machine learning.

## Project Overview
This project explores a large real-world body performance dataset (13,000+ records) to:
- Clean and preprocess data
- Visualize distributions and relationships
- Perform statistical analysis (t-tests, correlations)
- Build and evaluate a machine learning model to predict performance levels

## Folder Structure
- `data/` — Raw and processed data files (CSV, Excel)
- `src/` — Main analysis scripts (Python)
- `notebooks/` — Jupyter notebooks for interactive exploration
- `plots/` — Generated plots and images

## Features
- **Data Cleaning:** Handles missing values, encodes categorical features
- **Visualization:** Histograms, heatmaps, and more using matplotlib and seaborn
- **Statistical Analysis:** Group comparisons, t-tests, correlation matrices
- **Machine Learning:** Random Forest classifier for performance prediction (~75% accuracy)
- **Reproducibility:** All dependencies in `requirements.txt`, virtual environment support

## Getting Started

1. **Clone the repository**
   ```powershell
   git clone https://github.com/SUPRITJ/BODY-PERFORMANCE-ANALYSIS.git
   cd BODY-PERFORMANCE-ANALYSIS
   ```

2. **Set up a virtual environment**
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Run the main analysis**
   ```powershell
   python src/body_performance_analysis.py
   ```

5. **Explore interactively**
   ```powershell
   jupyter notebook notebooks/
   ```

## Results
- Prints summary statistics and model performance in the terminal
- Saves plots (distribution, correlation heatmap, confusion matrix) to the `plots/` folder

## Author
Suprit Jamdar