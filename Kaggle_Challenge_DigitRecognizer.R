---
  title: "Kaggle Competition"
author: "Ava Dossi"
date: "12/4/2021"
output: pdf_document
---
  
  ```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

#------------------------------------------------------#
## 1) Information
# Each *image*: height = 28 pixels, width = 28 pixels, total = 784 pixels  
# Each *pixel*: has a single pixel-value associated with it, indicating the lightness or darkness of that pixel --> with higher numbers meaning darke (pixel-value is an integer between 0 and 255, inclusive)  
# *training data*: 
# - 785 columns 
# - The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image. --> each row represents one image
# - Each pixel column in the training set has a name like pixelx, where x is an integer between 0 and 783, inclusive. To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero).

## 1.1) Libraries
library(devtools)
#install.packages("reticulate")
library(reticulate)
#install.packages("tensorflow")
library(tensorflow)
#install.packages("keras")
library(keras)


#------------------------------------------------------#
## 2) Data Insights
### 2.1) Read training data
train <- read.csv("data/train.csv", header=TRUE)
test <- read.csv("data/test.csv", header=TRUE)


### 2.2) Get overview of training data set
# short overview training data
head(train)
str(train)

# short overview test data
head(test)
str(test)

# means for each image
#rowMeans(train)


### 2.3) Visualize digits of data set
# visualize first 20 images of the digits from TRAINING data
for (i in 1:10) {
  # create a vector for each image not including the first column with the label information
  image_vec <- unlist(train[i,2:ncol(train)])
  
  # tranform vector to matrix
  image_matrix <- (matrix(image_vec,28,28, byrow=TRUE))
  
  # rotate matrix 
  rotate <- t(apply(image_matrix, 2, rev))
  
  # visualize number
  image(rotate ,col = grey(seq(0, 1, length = 256))) 
  Sys.sleep(1)
}


# visualize first 20 images of the digits from TEST data
for (i in 1:20) {
  # create a vector for each image not including the first column with the label information
  image_vec <- unlist(test[i,1:ncol(test)])
  
  # tranform vector to matrix
  image_matrix <- (matrix(image_vec,28,28, byrow=TRUE))
  
  # rotate matrix
  rotate <- t(apply(image_matrix, 2, rev))
  
  # visualize number
  image(rotate ,col = grey(seq(0, 1, length = 256)))
  Sys.sleep(1)
}
# 209037030352404331



#------------------------------------------------------#
## 3) Data preparation
### 3.1) Vector form
train_x <- as.matrix(train[,2:ncol(train)])
test_x <- as.matrix(test[,1:ncol(test)])


### 3.2) Rescaling of X (pixel)
# rescaling: from 0 - 255 to 0 - 1
train_x <- train_x / 255 # not the first colums (represents acutal digit)
test_x <- test_x / 255


### 3.4) One-hot encoded y vector (labels) - representation of categorical variables as binary vectors
# y data = 0 - 9 
# encode into binary class matrices with function to_categorical
train_y <- to_categorical(train[,1], 10)


### 3.5) EXTRA: Create extended data set by duplicating training data in a slightly shifted way
# create zero vectors to fill up image vector 
zero_col_train <- cbind(rep(0,nrow(train)), rep(0,nrow(train)), rep(0,nrow(train)))

# drop indices to indicate where to drop a value in image vector 
drop_index1 <- round(ncol(train)/2)
drop_index2 <- round(drop_index1/2)

# create new slightly shifted duplication of the original training data
train_x2 <- as.matrix(cbind(train[,3:drop_index2], 
                            train[,(drop_index2+2):drop_index1], 
                            train[,(drop_index1+2):(ncol(train))], 
                            zero_col_train)) # fill up end of vector according to number of dropped values 
# (= 3 zeros for each image vector)


# vector form for original data set
train_x <- as.matrix(train[,2:ncol(train)])

# merge both data sets (original and new one)
train_x_ext <- rbind(train_x,train_x2)

# rescale X (from 0 - 255 to 0 - 1)
train_x_ext <- train_x_ext / 255

# create output vector by dupiclating of the label of training data (order stayed the same --> we can attach it at the end)
train_y_ext <- to_categorical(c(train[,1],train[,1]), 10)


# # check beginning of the new sligthly shifted data set (digits are still readable)
# for (i in 1:10) {
#   # create a vector for each image not including the first column with the label information
#   image_vec <- unlist(train_x2[i,1:ncol(train_x2)])
#   
#   # tranform vector to matrix
#   image_matrix <- (matrix(image_vec,28,28, byrow=TRUE))
#   
#   # rotate matrix 
#   rotate <- t(apply(image_matrix, 2, rev))
#   
#   # visualize number
#   image(rotate ,col = grey(seq(0, 1, length = 256))) 
#   Sys.sleep(1)
# }




#------------------------------------------------------#
## 4) Model definition
# - sequential model (Layer after layer) 
# - pipe operator add layer  
# - input_shape for first layer --> specify input  
# - *relu = Rectified Linear Unit (ReLU)* --> if input negative => output = 0  
# - dropout layer randomly sets input units to 0 with frequency of rate at each step during training time, which helps prevent overfitting  
# - *softmax activation function = logistic function for last layer* 

# define model
model <- keras_model_sequential() # sequential model (Layer after layer) 
model %>% 
  layer_dense(units = 256, activation = "relu", input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = "softmax")

summary(model)


# - *Dense*: everything is connected  
# - *units* = dimensionality of the output space  
# - *activation* = relu, softmax, linear... etc...   
# - *input shape*: randomly setting a fraction rate of input units to 0at each update during training time, which helps prevent overfitting.  
# - *softmax*: logistic function  
# - compile() function to configure learning process before training a model  
# - *loss*: "categorical_crossentropy"  
# - *optimizer*: how the network is trained
# - what is shown but not trained on  


# add compiler flags
model %>% compile(
  optimizer = "rmsprop",             # network will update itself based on the training data & loss
  loss = "categorical_crossentropy", # measure mismatch between y_pred and y, calculated after each minibatch
  metrics = c("accuracy")            # measure of performace - correctly classified images
)



#------------------------------------------------------#
## 5) Model training, validation, test
# - train model using fit function  
# - *epochs* = amount of times repeated  
# - validation_split separates portion of training data into validation dataset and evaluates performance of model on validation dataset each epoch  

# Train model (with extended training data)
set.seed(1)
history <- model %>% fit(
  train_x_ext, train_y_ext, # with normal data set would be: train_x, train_y
  epochs = 60, batch_size = 128, 
  validation_split = 0.2
)
plot(history)

# check generate prediction on old data
# as.numeric(model %>% predict(train_x) %>% k_argmax())



#------------------------------------------------------#
## 6) Model prediction
# generate prediction on new data
as.numeric(model %>% predict(test_x) %>% k_argmax())

# correct test_x example data: 209037030352404331



#------------------------------------------------------#
## 7) Submission file
# colums
ImageId <- 1:nrow(test_x)
Label <- as.numeric(model %>% predict(test_x) %>% k_argmax())

# data frame
pred_dat <- data.frame(cbind(ImageId,Label))

# check
head(pred_dat, 20)

# create csv file
write.csv(pred_dat, "Kaggle_Challenge_AvaDossi.csv", row.names = FALSE)
