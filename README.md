# 🩺 Cardiac Arrhythmia Classification using CNN

This project presents a deep learning pipeline to classify cardiac arrhythmias using the [UCI Arrhythmia Dataset](https://archive.ics.uci.edu/dataset/32/arrhythmia). The process involves advanced preprocessing, feature selection, handling of class imbalance, and training a 1D Convolutional Neural Network (CNN) to achieve high classification performance.

---

## 📁 Dataset
The dataset (`arrhythmia.data`) contains:
- **279 features** extracted from ECG signals
- **1 target column** (`class`) indicating arrhythmia type
- Missing values represented by `'?'`

---

## ⚙️ Project Pipeline

### 🔹 Step 0: Import Libraries
Includes tools for:
- Data preprocessing: `pandas`, `numpy`, `scikit-learn`
- Visualization: `matplotlib`, `seaborn`
- Oversampling: `SMOTE` from `imblearn`
- Deep learning: `tensorflow.keras`

### 🔹 Step 1: Load and Clean Data
- Load the dataset
- Replace `'?'` with NaN and convert all entries to numeric
- Remove rare classes (11, 12, 13)

### 🔹 Step 2: Handle Missing Values
- Use `SimpleImputer` with `mean` strategy to fill missing values

### 🔹 Step 3: Feature Preparation
- Split into features (`X`) and labels (`y`)
- Apply standard scaling (`StandardScaler`) to normalize features

### 🔹 Step 4: Feature Selection
- Fit a `RandomForestClassifier` to rank feature importance
- Select top **89 features** using `SelectFromModel`

### 🔹 Step 5: Handle Imbalanced Classes
- Apply **SMOTE** (Synthetic Minority Over-sampling Technique)
- Resample data to balance the class distribution

### 🔹 Step 6: CNN Input Preparation
- Reshape features into 3D format for CNN: `(samples, features, 1)`
- One-hot encode the class labels

### 🔹 Step 7: Split Dataset
- Stratified train-test split (80% train / 20% test)

### 🔹 Step 8: Build the CNN Model
The model includes:
- Two `Conv1D` layers (64 and 128 filters)
- `MaxPooling1D`, `BatchNormalization`, `Dropout`
- A dense fully connected layer and final `softmax` output layer

### 🔹 Step 9: Train the Model
- Optimizer: `adam`
- Loss: `categorical_crossentropy`
- Trained for 60 epochs with a 20% validation split

### 🔹 Step 10: Evaluate Performance
- Evaluate on the test set
- Print **accuracy**, **classification report**, and **confusion matrix**

### 🔹 Step 11: Visualize Metrics
- Plot training vs. validation accuracy and loss over epochs

---

## ✅ Results

- 📈 **Final Test Accuracy:** `99.06%`
- 🔁 Balanced class distribution after SMOTE
- 🧠 Strong performance across multiple arrhythmia types

---

## 📊 Sample Output


---

## 🧠 Technologies Used

- **Python**
- **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**
- **scikit-learn**, **imblearn**
- **TensorFlow / Keras**

---

## 📦 Setup Instructions

```bash
git clone https://github.com/your_username/your_repo_name.git
cd your_repo_name

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

## 🧬 CNN Model Architecture

- Two convolutional blocks:
  - Conv1D → BatchNorm → MaxPooling → Dropout
- Fully connected Dense layer with dropout
- Output layer with Softmax activation for multiclass classification

## Authors
Riddhi Tanwani
Shriya Bajpai
Contributions Welcome!!

##If you find this project useful, please star the repository and share it!
