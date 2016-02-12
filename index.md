# Practical Machine Learning Course Project
Jessica Ramos  
February 11, 2016  

## Introduction

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

In this project, I will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. We made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg).

## Data
The training data for this project are available here:
    https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:
    https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

### Citation
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

Read more: http://groupware.les.inf.puc-rio.br/har#wle_paper_section#ixzz3zvU7UUeS


```r
#import datasets
trainData <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
                      na.strings = c("#DIV/0!", "NA", ""))
testData <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
                     na.strings = c("#DIV/0!", "NA", ""))
```

###Processing and Cleanup
While evaluating missing data, I found that either the columns did not have any NAs present or they had more than 97% NA.  I dropped these columns that contained more than 97% missing data.  These variables are not valuable in making the predictions.  




```r
#remove columns from datasets where columns contain NAs
trainData <- trainData[, colSums(is.na(trainData)) == 0]
testData <- testData[, colSums(is.na(testData)) == 0]
```


![](index_files/figure-html/unnamed-chunk-4-1.png)

Note that the `echo = FALSE` parameter was added to the code chunk to prevent printing of the R code that generated the plot.
