# Background on Imbalanced Data Sets

### Introduction

When we talk about imbalanced data sets in machine learning we are usually talking about situations were the _label_ or _classes_ of a dataset are not distributed equally. A very common example in Machine Learning (ML) literature is if we want to predict Postivie Cancer diagnosis. Here we have two classes: Postive Cancer Diagnosis, (Cancer) and Negative Cancer Diagnosis (No Cancer). In such data sets, the number of positive cancer diagnosis are much less than negative or non-cancer diagnosis -- the data set classes are imbalanced.

### Majority and Minority Classes

**Majority class** - is the more common label in a class-imbalanced dataset. For example, given a dataset containing 99% negative labels and 1% positive labels, the negative labels are the majority class, just like our cancer diagnosis example

**Minority class** - is the less common label in a class-imbalanced dataset. For example, given a dataset containing 99% negative labels and 1% positive labels, the positive labels are the minority class.

### Why Do we Care?

If we just apply imbalanced data directly into a machine learning algorithm, we are going to run into problems. The model no doubt will ignore the minority class as there are so few examples to train on in comparison to the majority class, so expect incorrect predictions for the underrepresented class. These erroneous predictions are especially bad if what we really care about predicting correctly is the minority class like in our cancer diagnosis example.

Imbalanced datasets are especially problematic for classification problem evaluations when using the accuracy metric of a [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix).

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

Where:

- True Positives (TP): The number of instances correctly predicted as positive.
- True Negatives (TN): The number of instances correctly predicted as negative.
- False Positives (FP): The number of instances incorrectly predicted as positive (Type I error).
- False Negatives (FN): The number of instances incorrectly predicted as negative (Type II error).

or expressed simply

$$Accuracy = \frac{NumberOfCorrectPredictions}{NumberOfTotalPredictions}$$

In the case of cancer diagnosis detection, the model would predict non-cancer most of the time with high accuracy, but it would be useless because what we really care about is detecting the minority cases accurately.

Other well known measures like:

- Area Under the Receiver-Operating Characteris (ROC) Curve [(AUC)](https://developers.google.com/machine-learning/glossary#AUC),
- [F1-Score](https://en.wikipedia.org/wiki/F-score), and
- [Precision-Recall Curves](https://www.geeksforgeeks.org/machine-learning/precision-recall-curve-ml/)

would be better measures for imbalanced data sets.

Ergo, it is important to understand how to identify, understand, and handle imbalanced problems.

### Known Techniques to Handle Imbalanced Data Sets

#### Collecting a bigger sample

Simply go out and get more data. This technique applies only if you are able to do so, but could be very simple.

#### Oversampling

Adds examples of the minority class to balance the data set (i.e., add more examples). In other words, we randomly duplicate observations of the minority class. The problem with this approach is that it leads to overfitting because the model learns from the same examples. Some techniques below help out here.

- **Simple random oversampling:**
  Random sampling with replacement from the minority class

- **Oversampling with shrinkage:**
  Random sampling, adding some noise/shrinkage to disperse the new samples. Called shrinkage because. Shrinkage
  shrinkage is the reduction in the effects of sampling variation.

  [src: Detailed Description Below](https://scisimple.com/en/keywords/shrinkage--k3qgv55)
  Shrinkage is a technique used in statistics to improve estimates by reducing the impact of random noise or errors. When working with data, especially when we have small samples, our estimates can be unreliable. Shrinkage helps by pulling these estimates closer to a certain value, which can make them more accurate.

  Shrinkage combines two types of estimates: one that is very precise but may vary a lot (high-variance) and another that is less precise but more stable (low-bias). By blending these two, we can get a better estimate that balances both qualities. This is useful in many statistical methods where accuracy is important.

  Shrinkage techniques are used in various fields. For example, they help in estimating relationships in complex data, like predicting extreme values or understanding patterns in large sets of information. It can also improve the estimation of characteristics in situations where data is limited or noisy, such as in finance or astronomy.

  The main benefit of shrinkage is better accuracy in estimates, especially when dealing with small amounts of data. It reduces the chances of making large errors by stabilizing our guesses. This approach can lead to clearer insights and more reliable predictions.

- **Oversampling using SMOTE:**
  Synthesize new samples based on the minority class. [Synthetic Minority Over-Sampling TEchnique] (SMOTE)(https://medium.com/@corymaklin/synthetic-minority-over-sampling-technique-smote-7d419696b88c) is a preprocessing technique used to address a class imbalance in a dataset.

  When acquiring more data isn’t an option, we have to resort to down-sampling or up-sampling. Down-sampling is bad because it removes samples that could otherwise have been used to train the model. Up-sampling on its own is less than ideal since it causes our model to overfit. SMOTE is a technique to up-sample the minority classes while avoiding overfitting. It does this by generating new synthetic examples close to the other points (belonging to the minority class) in feature space

  How SMOTE works:

  We repeat the process N/100 times. In other words, if the amount of over-sampling needed is 300%, only 300/100 = 3 neighbors from the k = #Some Number nearest neighbors are chosen and one sample is generated in the direction of each. In this case, we set N = 300. Therefore, we consider 3 random nearest neighbors of each point.

  - Take difference between a sample and its nearest neighbour
  - Multiply the difference by a random number between 0 and 10
  - Add this difference to the sample to generate a new synthetic example in feature space
  - Continue on with next nearest neighbour up to user-defined number (Here, N=300)

#### Undersampling

Removes examples of the majority class to balance the data set (i.e., reduce examples). In other words, we randomly sample observations of the majority class equal to size of the minority class. The problem with this approach is that it may mean that we remove useful information from the dataset.

- **Simple random undersampling:**
  Here, I just simple pandas library to perform sampling, then repeat the process again using the `imbalanced-learn` library.

- **Undersampling using K-Means:**

  Besides random sampling, we could also use the cluster centroid of the K-Means method as the new sample of the majority class. This means the new sample of the majority class is not the original data anymore. They are synthesized with cluster centroids. So the new sample should be more representative of the actual majority class data.

- **Undersampling using Tomek links:**

  A [Tomek link](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis) removes unwanted overlap between classes where majority class links are removed until all minimally distanced nearest neighbor pairs are of the same class.

  It is between two samples of different classes. When the two samples are the nearest neighbors of each other, they form a Tomek link.

  In our example of the binary classification problem, a Tomek link is a pair of examples from each class that is the closest neighbor across the dataset. After detecting such a link, we could remove data within the pair. Usually, we remove the sample from the majority class to achieve undersampling, i.e., remove the majority class close to the minority class. This removes ambiguity between the two classes.

  So, undersampling with Tomek links clean up the overlaps between classes, making them easier to distinguish.

#### Combining Oversampling and Undersampling

Looking at combining techniques to get better classifier model performance

#### Weighing Classes differently

Don't do oversampling or undersampling, weigh the different classes to guide the learning model creation.

#### Changing Algorithms

Here think about using other algorithms like decision trees, random forests, or XGBoost that by their very nature mitigate imbalanced data sets.

Ultimately, there is a fair amount of trial and error examing different techniques and evaluating models generated to see they achieve the desired performance at the appropriate resource costs!
