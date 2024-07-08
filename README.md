
# SMS Spam Classification Model

Welcome to the SMS Spam Classification project! This repository contains the code and resources for building a machine learning model to classify SMS messages as either "spam" or "genuine." This project was developed as part of my internship with Bharath Intern in the domain of Data Science.

## Table of Contents
* [Introduction](#introduction)
* [Dataset](#dataset)
* [Preprocessing](#preprocessing)
* [Models](#models)
* [Evaluation](#evaluation)
* [Results](#results)
* [Usage](#usage)
* [Future Work](#future-work)
* [Contributing](#contributing)
* [License](#license)

## Introduction
The goal of this project is to develop a model that can accurately classify SMS messages as spam or genuine. Spam messages can be a nuisance and even a security threat, so having an effective classification system is essential for filtering out unwanted messages.

## Dataset
The dataset used in this project is the SMS Spam Collection Data Set, which contains a collection of SMS messages labeled as either spam or ham.

* **Source**: UCI Machine Learning Repository
* **Format**: CSV file with columns `label` and `message`
   * `label`: Indicates whether the message is spam or genuine
   * `message`: The text of the SMS message

## Preprocessing
Preprocessing steps include:

1. **Removing Punctuation**: Eliminates unnecessary punctuation marks.
2. **Converting to Lowercase**: Standardizes the text to lowercase.
3. **Removing Digits**: Strips out numeric characters.

## Models
The following machine learning algorithms were used to build the classification models:

* **Naive Bayes**
* **Decision Tree**
* **Random Forest**
* **K-Nearest Neighbors (KNN)**

## Evaluation
The models were evaluated using:

* **Train-Test Split**: Split the dataset into training, validation, and test sets.
* **TF-IDF Vectorization**: Transformed text data into numerical features.
* **K-Fold Cross-Validation**: Employed to ensure model reliability.

Evaluation metrics include:

* **Accuracy**
* **Precision**
* **Recall**
* **Classification Report**

## Results
Each model's performance was assessed based on accuracy, precision, and recall. The results of the cross-validation and validation steps guided the selection of the best-performing model.

## Usage
To use this project, follow these steps:

1. **Clone the repository**
```bash
  git clone https://github.com/Nandakishor-P/Task1-SMS-Classifier
```
2. **Install dependencies**
```bash
install pandas scikit-learn numpy
```
3. **Run the script**
```bash
python train.py
```
4. **Load and use the saved models**
```bash
python load_and_classify.py
```

# Note 
If python is not working try python3
## Future Work
Potential improvements and future work include:

* Expanding the dataset for better generalization.
* Exploring more advanced NLP techniques and models.
* Integrating the model into a real-time SMS filtering application.

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, feel free to submit a pull request or open an issue.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
```

