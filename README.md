
# Sentiment Analysis Model

## Overview

This sentiment analysis model aims to classify the sentiment of customer reviews as positive or negative based on their textual content. The model utilizes a variety of machine-learning algorithms and text processing techniques to achieve accurate sentiment classification.

## Features

- **Text Preprocessing**: The text data is preprocessed to remove non-alphabet characters, convert to lowercase, and perform stemming to standardize the text format.
- **Word Cloud Visualization**: Word clouds are generated to visualize the frequency of positive and negative words in the customer reviews.
- **Bag of Words Representation**: The CountVectorizer is used to create a bag of words representation of the text data, capturing the frequency of words in the corpus.
- **Model Selection**: Multiple machine learning algorithms including Random Forest, XGBoost, and Decision Trees are experimented with to determine the best-performing model for sentiment classification.
- **Model Evaluation**: The models are evaluated using training and testing accuracy, confusion matrix, and cross-validation to ensure robust performance and generalization to unseen data.
- **Hyperparameter Tuning**: Grid Search is applied to optimize the hyperparameters of the selected models, improving their performance further.

## Usage

1. **Data Preprocessing**: Ensure that the text data is preprocessed using the provided preprocessing steps to prepare it for model training.
2. **Model Training**: Train the sentiment analysis model using the preprocessed text data and selected machine learning algorithms.
3. **Model Evaluation**: Evaluate the trained models using appropriate metrics such as accuracy, confusion matrix, and cross-validation scores to assess their performance.
4. **Hyperparameter Tuning**: Optionally, perform hyperparameter tuning using Grid Search to optimize the model's hyperparameters for better performance.
5. **Model Persistence**: Save the trained models and preprocessing objects for future use and deployment in production environments.

## Model Accuracies

- **Random Forest**:
   - Training Accuracy: 99.46%
   - Testing Accuracy: 94.50%
   - Cross Validation Mean Accuracy: 93.38%

- **XGBoost**:
   - Training Accuracy: 97.14%
   - Testing Accuracy: 94.18%

- **Decision Tree Classifier**:
   - Training Accuracy: 99.46%
   - Testing Accuracy: 92.80%

## Dependencies

- Python 3.x
- scikit-learn
- pandas
- matplotlib
- seaborn
- wordcloud
- xgboost

## License

This sentiment analysis model is licensed under the [MIT License](LICENSE).


