# 🏥 Liver Cirrhosis Stage Detection

A machine learning web application that **predicts liver cirrhosis stages** using clinical biomarkers and patient medical data. This tool assists healthcare professionals in early detection and risk assessment of liver disease progression.

![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=flat&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.3.3-000000?style=flat&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-F7931E?style=flat&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Features](#-features)
- [About Liver Cirrhosis](#-about-liver-cirrhosis)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Model Details](#-model-details)
- [Input Parameters Guide](#-input-parameters-guide)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)

---

## ✨ Features

- 🎯 **Real-time Predictions** - Get instant liver cirrhosis stage predictions
- 📊 **Multiple Input Methods**:
  - Manual form entry for single patient
  - Batch file upload (CSV/XLSX) for multiple patients
  - REST API for programmatic access
- 📈 **Probability Scores** - Confidence levels for each stage classification
- 🎨 **User-Friendly Interface** - Clean, responsive web UI
- 🔒 **Privacy-First** - Runs locally, no data sent to external servers
- 📱 **Clinical Biomarkers** - Based on standard liver function tests

---

## 🏥 About Liver Cirrhosis

Liver cirrhosis is the final stage of liver disease, characterized by severe scarring and loss of liver function. Early detection and staging are crucial for:

- **Treatment Planning** - Appropriate intervention based on disease stage
- **Risk Assessment** - Identifying patients requiring urgent care
- **Prognosis Estimation** - Understanding disease progression

### Disease Stages

| Stage | Classification | Clinical Significance |
|-------|---|---|
| **1** | Early Stage | Mild liver damage, close monitoring recommended |
| **2** | Moderate Stage | Noticeable liver impairment, active treatment needed |
| **3** | Advanced Stage | Significant liver damage, intensive care required |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Dharmik1602/Liver_Cirrhosis_Stage_Detection.git
   cd Liver_Cirrhosis_Stage_Detection
