
# PAD ML Workflow using v2 API

[![PAD API v2 Documentation](https://img.shields.io/badge/PAD%20API%20v2-Documentation-blue?logo=swagger)](https://pad.crc.nd.edu/docs)

This repository provides a complete workflow for exploring and training machine learning models using data from the **PAD API v2**. The notebooks will guide you through each step, from setting up your environment to evaluating your trained models.

---

## 1. Python Basics for PAD Project Data Exploration

Before diving into the machine learning pipeline, it's important to understand the Python fundamentals and how to explore data from the PAD API. This notebook introduces you to:
- Python basics (variables, functions, loops, etc.).
- Working with **Pandas** for data manipulation.
- Reading and saving CSV files.
- Exploring data from PAD projects.

You can find the notebook here:  [**Python Basics for PAD Project Data Exploration**](https://colab.research.google.com/drive/1CWoDaFxGord3w60mJg7t2pufraqXhdH-)

[![**Python Basics for PAD Project Data Exploration**](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CWoDaFxGord3w60mJg7t2pufraqXhdH-)


## 2. Setting Up Your Environment in Google Colab

To run the notebooks efficiently, we recommend using **Google Colab** along with **Google Drive** for storage. This environment allows you to utilize GPU/TPU resources for model training and manage data files directly from your Google Drive.

**Open the Pre-configured Notebook**:  
Access the notebook by clicking the link below. This notebook will guide you through mounting Google Drive and setting up the necessary packages for the workflow.
   
You can find the notebook here: [**Google Colab Initial Setup**](https://colab.research.google.com/drive/1fsHOC4YHwLNRNn64ymHitD_iW6JPFiMh)
   
[![**Google Colab Initial Setup**](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fsHOC4YHwLNRNn64ymHitD_iW6JPFiMh)

   

## 3. Explore Data Available via PAD API

The PAD API provides access to rich datasets that you can explore and prepare for machine learning tasks. The following notebooks will help you interact with the API, fetch data, and prepare it for machine learning workflows.

**Exploration Notebook**: Use this notebook to explore the available data via the PAD API. You can visualize and understand the structure of the data, analyze metadata, and assess data quality.

You can find the notebook here: [**Exploration Notebook in Google Colab**](https://colab.research.google.com/drive/12ydoCcnnwWkyBQsO3LOe70ezCB9QuB-0) 

[![**Exploration Notebook in Google Colab**](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12ydoCcnnwWkyBQsO3LOe70ezCB9QuB-0)  


## 4. ML Pipeline Notebooks

The core of this repository includes notebooks to guide you through building, training, and evaluating machine learning models using PAD API data. Each stage of the ML pipeline is covered in detail, from data preprocessing to model evaluation.

### Key Notebooks in the ML Pipeline:

- **Data Preparation Notebook**:
  This notebook will help you clean, transform, and prepare your data for model training. Tasks include handling missing values, feature selection, and normalization.

  [**Data Preparation Notebook**](link_to_data_preparation_notebook)

- **Create Dataset**:  
  Learn how to create and organize datasets for machine learning models. You will work with the processed data from the previous notebooks and set up your training and testing datasets.

  [**Create Dataset Notebook**](link_to_create_dataset_notebook)

- **Training Notebook**:  
  Use this notebook to define, train, and fine-tune your machine learning models. The notebook includes examples for building models using libraries such as `scikit-learn`, `tensorflow`, or `keras`.

  [**Training Notebook**](link_to_training_notebook)

- **Evaluation Notebook**:  
  After training your model, use the evaluation notebook to measure its performance using appropriate metrics. Visualize model accuracy, confusion matrices, and other performance indicators.

  [**Evaluation Notebook**](link_to_evaluation_notebook)

---

## 5. Workflow Summary

This repository offers a comprehensive set of tools and notebooks that cover the following steps:
- **Data Exploration**: Understand the data using Pandas and visualization tools.
- **Data Preparation**: Clean and transform the data for modeling.
- **Model Training**: Train machine learning models on the prepared dataset.
- **Evaluation**: Evaluate model performance and make adjustments as needed.

Each notebook in this repository is designed to work together as part of a smooth, integrated pipeline. You can easily modify any step of the process to suit your specific needs.

---

Feel free to explore the [**PAD API v2 Documentation**](https://pad.crc.nd.edu/docs) to learn more about the API and available data.

---

## Next Steps
Once you’ve completed the notebooks provided, you can:
- Experiment with different model architectures.
- Adjust hyperparameters for better performance.
- Extend the workflow to include more advanced techniques like transfer learning or model tuning.


  

