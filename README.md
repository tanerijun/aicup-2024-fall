# AICUP 2024 Fall Solution

```
AI CUP 2024 秋季賽
根據區域微氣候資料預測發電量競賽報告
```

Competition details:
https://tbrain.trendmicro.com.tw/Competitions/Details/36

### Dependencies
- numpy v1.26.4
- pandas v2.2.2
- matplotlib v3.9.2
- seaborn v0.13.2
- scikit-learn v1.5.1
- pytorch v2.5.1
- joblib v1.4.2

### Folder structure
```
Parent
  |__ data
      |__ L1_Train.csv
      |__ ...
      |__ weather.csv (Data from external API)
      |__ upload.csv (Data we have to fill and submit to AICUP)
  |__ aicup_2024_fall_solution.ipynb
```

Note that the training data is not provided in this repo. Please consult the [official site](https://tbrain.trendmicro.com.tw/Competitions/Details/36) for the data.

## Feature Engineering

Through data exploration and descriptions provided by AICUP organizer. It's clear that most of the features are pretty unreliable (sensor issue, sunlight cap, etc.). Furthermore, these features are not provided for us in `upload.csv`. Therefore, instead of taking the risk and predicting these unreliable features for the queries in `upload.csv`, we decided to leverage external weather info from [CODIS](https://codis.cwa.gov.tw/).

We've downloaded and processed the relevant data. It's provided in the form of a csv file in `data/weather.csv`.

## Model Choice Reasoning

We can clearly see from data exploration step in the notebook that this is not your trivial time-series predicting problem. Instead of predicting the future, we're doing the classic regression problem, `fill-in-the-blank`, here. Therefore, we decided to use the classic and proven Feed-Forward Neural Network (FNN) for this task.

Note that we've also tried other regression model like Random Forest, XGBoost, LightGBM, CatBoost, etc. But our experiment suggests that classic FNN is better for this task.

We've also experimented with Transformer, and the result is only slightly below classic FNN, so there's definetely potential using this approach. We ended up having to give up on this approach due to time constraint.

## Model Architecture

### Overview
Key components:
- Squeeze-and-Excitation (SE) Block
- Multi-scale Feature Extraction
- Pyramid Structure
- Skip Connections with Gating
- Multi-head Prediction

#### A. Squeeze-and-Excitation Block
- Reduces dimensionality by half and then restores it
- Uses GELU activation and Sigmoid for feature recalibration

#### B. Multi-scale Feature Extraction (Three Parallel Paths)
- Deep path (1024 units)
- Medium path (512 units)
- Short path (256 units)

#### C. Pyramid Structure
Three layers with decreasing dimensions:
- Layer 1: 1792 → 896
- Layer 2: 896 → 448
- Layer 3: 448 → 224

Each layer includes:
- Linear transformation
- Layer normalization
- GELU activation
- Dropout (decreasing rates: 0.25, 0.2, 0.15)

#### D. Gating Mechanism
Three gates corresponding to pyramid layers:

- Gate 1: 1792 → 896
- Gate 2: 896 → 448
- Gate 3: 448 → 224

Each gate uses Sigmoid activation for feature selection

#### E. Output Heads
Three parallel output heads
Each head: 224 → 64 → 1

### Training Parameters

#### Optimizer
Optimizer: `AdamW`
learning_rate: `0.001`
weight_decay: `1e-5`

#### Learning Rate Scheduler
Scheduler: OneCycleLR
max_lr: `0.001`
epochs: `100`,
pct_start: `0.3`

#### Training Configuration
- Epochs: 100
- Loss Function: MSE Loss
- Early Stopping Patience: 20
- Gradient Clipping: max_norm=1.0