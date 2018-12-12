# Problem 2

# 1. drugName (categorical): name of drug 
# 2. condition (categorical): name of condition 
# 3. review (text): patient review 
# 4. rating (numerical): 10 star patient rating 
# 5. date (date): date of review entry 
# 6. usefulCount (numerical): number of users who found review useful
# 
# train_set(75%), test_set(25%)

# Use Reference in the Youtube Series created by Data Science Dojo
# setwd('/Users/jieyichen/Desktop/R Final Project/')
# linear model and neural networks

require(data.table)
drug_test = as.data.frame(fread('drugsComTest_raw.tsv'))
drug_train = as.data.frame(fread('drugsComTrain_raw.tsv'))

library(ggplot2)

data_preprocess = function(dataframe){
  # check whether there are missing values in the data
  if (length(which(!complete.cases(dataframe))) == 0) {
    print('Dataset has no missing values.')
  } else{
    print('Dataset has missing values.')
  }
  print('The table below shows the rating score distribution:')
  # see the rating score distribution
  print(prop.table(table(dataframe$rating)))
  # get the review length
  dataframe$review_length = nchar(dataframe$review)
  return(dataframe)
}

ggplot_draw = function(dataframe){
  # visualize the review lengths with class labels
  ggplot(dataframe, aes(x = rating, fill = review_length)) + 
    geom_histogram(bins = 20) + 
    labs(y = 'Review Text Count', x = 'Length of Text', title = 'Distribution of Review Text Lengths with Rating Score') +
    theme(plot.title = element_text(hjust = 0.5))
}

drug_train_output = data_preprocess(drug_train)
ggplot_draw(drug_train_output)
drug_test_output = data_preprocess(drug_test)
ggplot_draw(drug_test_output)

# We have checked that the data distribution is relatively similar between the train set and test set.


# Data Pre-Processing Steps:
# 1) Tokenize the words, remove numbers, punctuations, symbols and hyphens
# 2) Covert the case of tokens so that we do not have repetitive words 
# 3) Filter out the stop words (common words that provides very little meaning, such as 'the', 'a' and etc)
# 4) Perform stemming (take similar words and collapse them into one) on the tokens, 
# 5) Create a matrix using the pre-processed tokens


#install.packages('quanteda')
library(quanteda)

word_tokenize = function(dataframe) {
  # tokenize the words
  # remove numbers, punctuations, symbols and hyphens because we only want to analyze the feelings that the text tries to convey
  dataframe_tokens = tokens(dataframe$review, what = 'word', remove_numbers = TRUE, remove_punct = TRUE, remove_symbols = TRUE, remove_hyphens = TRUE)
  # lower case the tokens
  dataframe_tokens = tokens_tolower(dataframe_tokens)
  # remove stop words
  dataframe_tokens = tokens_select(dataframe_tokens, stopwords(), selection = 'remove')
  # stem the tokens
  dataframe_tokens = tokens_wordstem(dataframe_tokens, language = 'english')
  list_text = sapply(dataframe_tokens, as.character)
  return(list_text)
}

test_text = word_tokenize(drug_test)
train_text = word_tokenize(drug_train)

#install.packages('sentimentr')
library(sentimentr)

sentiment_avg = function(text_analyze) {
  # gives a score that indicates the positiveness or negativeness of the words
  text_sentiment = sentiment(text_analyze)
  # take the average of the sentiment score
  return (mean(text_sentiment$sentiment))
}

# We found that the review messages generally converys negative feelings because the patient describes
# their pains caused by the disease in the review as well. 

create_feature_df = function(sentiment_score, orig_df) {
  # create sentiment_df with its score
  final_df = data.frame(sentiment_score)
  # select certain columns from df
  orig_df_select = orig_df[, c('drugName', 'condition', 'rating', 'usefulCount', 'review_length')]
  # combine sentiment_df with selected_column_df
  drug_df = cbind(final_df, orig_df_select)
  drug_df$drugName = as.numeric(as.factor(drug_df$drugName))
  drug_df$conditionid = as.numeric(as.factor(drug_df$condition))
  drug_df = drug_df[,c('conditionid', 'drugName', 'rating', 'sentiment_score', 'usefulCount', 'review_length')]
  return(drug_df)
}

mean_rating = function(train_list, drug_train_output){
  train_df = create_feature_df(train_list, drug_train_output)
  # get the mean and sd value
  dt = data.table(train_df)
  dt_table = dt[, list(mean = mean(sentiment_score), sd = sd(sentiment_score)), by = rating]
  
  return (dt_table[with(dt_table, order(-mean))])
}

set.seed(9999)
trainidxs <- sample(1:nrow(drug_train),1200)
testidxs <- sample(1:nrow(drug_test),400)
trainset <- drug_train[trainidxs, ]
testset <- drug_test[testidxs, ]

train_list = sapply(train_text[trainidxs], sentiment_avg)
test_list = sapply(test_text[testidxs], sentiment_avg)
mean_rating(train_list, drug_train_output[trainidxs, ])
mean_rating(test_list, drug_test_output[testidxs, ])


library(caret)

convert_df = function(df_list, drug_train){
  df = create_feature_df(df_list, drug_train)
  # the range of the variable is relatively large, therefore, we need to normalize the data
  # normalize the data in interval [0, 1]
  df$usefulCount = df$usefulCount + 0.1
  
  aggre_sum_condition = aggregate(.~conditionid, df, sum)
  aggre_sum_condition = aggre_sum_condition[, c('conditionid', 'usefulCount')]
  colnames(aggre_sum_condition) = c('conditionid', 'usefulCount_condition_sum')
  df = merge(df, aggre_sum_condition, by = 'conditionid', all = T)
  
  aggre_sum_drug = aggregate(.~drugName, df, sum)
  aggre_sum_drug = aggre_sum_drug[, c('drugName', 'usefulCount')]
  
  colnames(aggre_sum_drug) = c('drugName', 'usefulCount_drug_sum')
  
  df = merge(df, aggre_sum_drug, by = 'drugName', all = T)
  
  df$uWeight_condition = df$usefulCount / df$usefulCount_condition_sum
  df$uWeight_drug = df$usefulCount / df$usefulCount_drug_sum
  
  df = df[, c('conditionid',  'drugName',  'rating',  'sentiment_score', 'uWeight_condition', 'uWeight_drug', 'review_length')]
  
  return(df)
}


train_final_df = convert_df(train_list, drug_train_output[trainidxs, ])
test_final_df = convert_df(test_list, drug_test_output[testidxs, ])


RStoReg <- function (ratingsIn, useNij = FALSE, weighted = FALSE) 
{
  UINN <- getUINN(ratingsIn, weighted)
  if (ncol(ratingsIn) > 3) {
    covs <- as.matrix(ratingsIn[, -(1:3)])
    dimnames(covs)[[2]] <- names(ratingsIn[, -(1:3)])
  }
  usrsInput <- as.character(ratingsIn[, 1])
  itmsInput <- as.character(ratingsIn[, 2])
  userMeans <- UINN$uMeans[usrsInput]
  itemMeans <- UINN$iMeans[itmsInput]
  means <- data.frame(uMeans = userMeans, iMeans = itemMeans)
  if (useNij) {
    userN <- UINN$uN[usrsInput]
    itemN <- UINN$iN[itmsInput]
    means <- cbind(means, userN, itemN)
    names(means)[3:4] <- c("uN", "iN")
  }
  if (ncol(ratingsIn) > 3) {
    xy <- cbind(means, covs)
  }
  else xy <- means
  xy <- cbind(xy, ratingsIn[, 3])
  names(xy)[ncol(xy)] <- names(ratingsIn)[3]
  rownames(xy) <- rownames(ratingsIn)
  xy
}

getUINN <- function (ratingsIn, weighted = FALSE) 
{
  users <- as.character(ratingsIn[, 1])
  items <- as.character(ratingsIn[, 2])
  
  if(weighted) {
    ratings1 <- ratingsIn$sentiment_score * ratingsIn$uWeight_condition
    ratings2 <- ratingsIn$sentiment_score * ratingsIn$uWeight_drug
    umean <- function(x) {
      sum(x)
    }
    Ni. <- tapply(ratings1, users, length)
    N.j <- tapply(ratings2, items, length)
    usrMeans <- tapply(ratings1, users, umean)
    itmMeans <- tapply(ratings2, items, umean)
    final <- list(uMeans = usrMeans, iMeans = itmMeans, uN = Ni., iN = N.j)
    return (final)
  } else {
    ratings1 <- ratingsIn$sentiment_score
    ratings2 <- ratingsIn$sentiment_score
    umean <- function(x) {
      mean(x)
    }
    Ni. <- tapply(ratings1, users, length)
    N.j <- tapply(ratings2, items, length)
    usrMeans <- tapply(ratings1, users, umean)
    itmMeans <- tapply(ratings2, items, umean)
    final <- list(uMeans = usrMeans, iMeans = itmMeans, uN = Ni., iN = N.j)
    return (final)
  }
}


model_list = list()
# Linear Model
# 1)	Yij ~ rev_score + eij
library(rectools)
udconv_train = RStoReg(train_final_df, weighted = FALSE)
# run the linear model
delete = c('uWeight_condition', 'uWeight_drug')
udconv_train = udconv_train[, !(colnames(udconv_train) %in% delete), drop=FALSE]
udconvout = lm(rating ~ ., data = udconv_train[-c(1:2)])
udconvout$coefficients
# predict the test_df using the linear model
udconv_test = RStoReg(test_final_df,weighted = FALSE)
preds_cov = predict(udconvout, udconv_test)
model_list[['linear_on_segment_score']] = mean(abs(preds_cov - udconv_test[, 'rating']), na.rm = T)


# 2) E(Yij|i,j) = u + Bj + eij
udconvout_1 = lm(rating ~ ., data = udconv_train[-1])
udconvout_1$coefficients
# predict the test_df using the linear model
udconv_test_1 = RStoReg(test_final_df,weighted = FALSE)
preds_cov_1 = predict(udconvout_1, udconv_test_1)
model_list[['linear_on_imeans']] = mean(abs(preds_cov_1 - udconv_test_1[, 'rating']), na.rm = T)

# 3)	E(Yij|i,j) = u + ai + Bj + eij
udconvout_2 = lm(rating ~ ., data = udconv_train)
udconvout_2$coefficients
# predict the test_df using the linear model
udconv_test_2 = RStoReg(test_final_df,weighted = FALSE)
preds_cov_2 = predict(udconvout_2, udconv_test_2)
model_list[['linear_on_umeans_imeans']] = mean(abs(preds_cov_2 - udconv_test_2[, 'rating']), na.rm = T)

# 4)	E(Yij|i,j) = u + ai + Bj + eij (Weighted)
udconv_train_3 = RStoReg(train_final_df, weighted = TRUE)
# run the linear model
delete = c('uWeight_condition', 'uWeight_drug')
udconv_train_3 = udconv_train_3[, !(colnames(udconv_train_3) %in% delete), drop=FALSE]
udconvout_3 = lm(rating ~ ., data = udconv_train_3)
udconvout_3$coefficients
# predict the test_df using the linear model
udconv_test_3 = RStoReg(test_final_df,weighted = TRUE)
preds_cov_3 = predict(udconvout_3, udconv_test_3)
model_list[['linear_on_umeans_imeans_weighted']] = mean(abs(preds_cov_3 - udconv_test_3[, 'rating']), na.rm = T)

model_list


# Neural Network
library(neuralnet)

# scale the data
scaledData <- scale(train_final_df)

# get all the column name of the dataframe
allVars = colnames(train_final_df)
# remove the "rating", "condition_id", "drugName" from all vars and the rest are predictor variables
predictorVars = allVars[!allVars%in% c("rating", "condition_id", "drugName", "uWeight_condition", "uWeight_drug")]

predictorVars = paste(predictorVars, collapse = "+")
form = as.formula(paste("rating~", predictorVars, collapse = "+"))

# 4 hidden nodes in the second layer and 2 hidden nodes in the third layer

neuralnet_cv = function(node_layer_1, node_layer_2){
  neuralModel = neuralnet(formula = form, hidden = c(node_layer_1, node_layer_2), linear.output = TRUE, data = scaledData, stepmax=1e8)
  plot(neuralModel)
  # Neural network produces continuous variable because our last layer uses linear output. 
  predictions = compute(neuralModel, test_final_df[, allVars[!allVars%in%c("rating", "condition_id", "drugName", "uWeight_condition", "uWeight_drug")]])
  scaledResults = predictions$net.result * attr(scaledData, "scaled:scale")["rating"]  + attr(scaledData, "scaled:center")["rating"]
  return(mean(abs(scaledResults - udconv_test[, 'rating'])))
}

# do cross validation to find the optimal hidden nodes for the neural network
hidden_node_df = data.frame(c(4, 2), c(5, 4), c(6, 6))
hidden_node_df = data.frame(t(hidden_node_df))
names(hidden_node_df) = c('nodes_layer_1', 'nodes_layer_2')

for (num_row in seq(1, nrow(hidden_node_df))) {
  node_layer_1 = hidden_node_df$nodes_layer_1[num_row]
  node_layer_2 = hidden_node_df$nodes_layer_2[num_row]
  hidden_node_df$error_rate[num_row] = c(neuralnet_cv(node_layer_1, node_layer_2))
}

rownames(hidden_node_df) = NULL
hidden_node_df
