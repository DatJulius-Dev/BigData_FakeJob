Trainer R:

# Input: dataset
# Output: model

# The code below is an example which can be replaced with your own code.
# See the help page of "Create R Model" module for the list of predefined functions and constants.

library(e1071)
features <- get.feature.columns(dataset)
labels   <- as.factor(get.label.column(dataset))
train.data <- data.frame(features, labels)
feature.names <- get.feature.column.names(dataset)
names(train.data) <- c(feature.names, "Class")

model <- naiveBayes(Class ~ ., train.data)


Scorer R:
# Input: model, dataset
# Output: scores

# The code below is an example which can be replaced with your own code.
# See the help page of "Create R Model" module for the list of predefined functions and constants.

library(e1071)
probabilities <- predict(model, dataset, type="raw")[,2]
classes <- as.factor(as.numeric(probabilities >= 0.5))
scores <- data.frame(classes, probabilities)


Python Script:
def azureml_main(dataframe):
    import matplotlib
    matplotlib.use("agg")
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    scores = dataframe.ix[:, ("fraudulent", "classes", "probabilities")]
    ytrue = scores["fraudulent"]
    ypred = np.array([float(val) for val in scores["classes"]])    
    probabilities = scores["probabilities"]
    
    accuracy, precision, recall, auc = \
    accuracy_score(ytrue, ypred),\
    precision_score(ytrue, ypred),\
    recall_score(ytrue, ypred),\
    roc_auc_score(ytrue, probabilities)
    
    metrics = pd.DataFrame();
    metrics["Metric"] = ["Accuracy", "Precision", "Recall", "AUC"];
    metrics["Value"] = [accuracy, precision, recall, auc]

# Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(ytrue, probabilities)
    fig = plt.figure()
    axis = fig.gca()
    axis.plot(fpr, tpr, linewidth=8)
    axis.grid("on")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("ROC Curve")
    fig.savefig("roc.png")

    return metrics,





