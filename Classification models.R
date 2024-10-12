#Classification Models

library(tidyverse) #for data cleaning
library(ggplot2) #t-test plot
#install.packages("rms")
library(rms) #tried to do log. reg. with lrm()
library(nnet) #multinomial log. reg. with multinom()
library(leaps) #for step function to get best variables
library(corrplot) #for correlation plot
library(ggcorrplot) #for correlation plot by factor
library(randomForest)
library(MASS) #LDA,QDA
library(e1071) #Naive Bayes
library(class) #KNN
library(caret) #to make k-folds

setwd("C:/Users/Kerstin/Desktop/GT/7406/Project")
data <- read.csv(file = "spotify_songs.csv", sep=",", header=TRUE);
head(data)
dim(data)

#########################################
#Check for null values in the dataframe
########################################

has_null_values <- any(is.na(data))

if (has_null_values) {
  print("The dataframe has null values.")
} else {
  print("The dataframe does not have null values.")
}

rows_with_null <- which(!complete.cases(data))

# Print rows with null values
if (length(rows_with_null) > 0) {
  print("Rows with null values:")
  print(rows_with_null)
} else {
  print("No rows have null values.")
}

#new dataframe without nulls
data2 <- data[-rows_with_null,]
dim(data2)

sum(duplicated(data2$track_id)) #4476 duplicate rows

data3 <- data2 %>% distinct(track_id, .keep_all=TRUE)
dim(data3)


#######################################
# data with 0 popularity
######################################

#there is a signicant amount of tracks with 0 popularity
hist(data3$track_popularity) #normally distributed
table(data3$track_popularity == 0) #2698 with 0 opoularity
2698/(30130+2698) #8.2% is a large amount

data_0_pop <- data3 %>% filter(track_popularity == 0)
(data_0_pop %>% select(track_name, track_artist))[1:50,]

data4 <- data3 %>% filter(track_popularity != 0)
dim(data4)
hist(data4$track_popularity)

#Getting rid of identical songs with different popularity, keeping the max value to reflect the song's true popularity
data_dup <- data4 %>% dplyr::select(track_name, track_artist, track_popularity) %>% group_by(track_name, track_artist) %>% summarise(n=n()) %>% filter(n > 1)
head(data_dup)
dim(data_dup) #1454 3


data5 <- data4
for (i in 1:1454) {
  sub_df <- data4 %>% filter(track_name == data_dup[i,]$track_name, track_artist == data_dup[i,]$track_artist) #[data4$track_name == data_dup[i,]$track_name,]
  del_rows <- sub_df[-which.max(sub_df$track_popularity), ]
  data5 <- data5 %>% filter(!(track_id %in% del_rows$track_id))
}
data5 %>% dplyr::select(track_name, track_artist, track_popularity) %>% group_by(track_name, track_artist) %>% summarise(n=n()) %>% filter(n > 1)
dim(data5) #23965 23


data4[data4$track_name == "'Till I Collapse",]
data5[data5$track_name == "'Till I Collapse",]

write.csv(data5, "spotify_clean.csv", row.names=FALSE)


data6 <- data5[order(data5$track_popularity),][2397:23965,]
dim(data6)
hist(data6$track_popularity)
sum(data5$track_popularity == 0)

write.csv(data6, "spotify_clean_top90.csv", row.names=FALSE)

#so I don't have to rerun everythin
data6 <- read.csv("spotify_clean_top90.csv", header=TRUE)
head(data6)
##########################################
# Variable Selection for Classification
##########################################
length(unique(data6$track_id)) == dim(data6)[1] #verifying that track ids are unique


names(data6)

data_class <- data6 %>% dplyr::select(track_id, playlist_genre, danceability:duration_ms)#, , energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms)
head(data_class)

#6 possible genres to classify
genres <- unique((data_class$playlist_genre))

#turning them into factors, track_id is the id col and the rest are numeric
data_class$playlist_genre <- as.factor(data_class$playlist_genre)
str(data_class)
head(data_class)
names(data_class)
num_cols <- names(data_class)[3:14]

class_formula <- paste("playlist_genre~",paste(num_cols, collapse="+"), sep="")
lrm_class <- lrm(as.formula(class_formula), data=data_class, x=T, y=T)
lrm_class$freq
table(data_class$playlist_genre)
lrm_class$coefficients
predict(lrm_class, data_class)
anova(lrm_class, test="LR") #key and loudness do not pass the chi sq test

#multinomial logistic regression via neural network
multi_class <- multinom(as.formula(class_formula), data=data_class)
#residual deviance compared to full model is 107932

#trying to see best variables

#this takes forever, probs since this isn't a simple linear model
#library(leaps)
#step(multi_class)

#end up with residual deviance: 60105.14
#AIC: 60225.14
#variables: danceability, energy, loudness, mode, speechiness, acousticness, instrumentalness
#liveness, valence, tempo, and duration_ms

#no key 
table(predict(multi_class, data_class)) #not bad
1 - mean(predict(multi_class, data_class) == data_class$playlist_genre)

multi_class2 <- multinom(playlist_genre~danceability+energy+loudness+mode+speechiness+acousticness+instrumentalness+liveness+valence+tempo+duration_ms, data=data_class)
table(predict(multi_class2, data_class))
1 - mean(predict(multi_class2, data_class) == data_class$playlist_genre)
table(data_class$playlist_genre)

#correlation

#library(corrplot)
#corrplot(cor(data_class[,3:12]), method="color")

#install.packages("ggcorrplot")
#library(ggcorrplot)
model.matrix(~0+., data=data_class[,-1]) %>% 
  cor(use="pairwise.complete.obs") %>% 
  ggcorrplot(show.diag=FALSE)

selected_vars <- c("danceability", "energy", "loudness", "mode", "speechiness", 
                   "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms")


################################
# Model Building
#################################

data_class_final <- data_class[,c(-5)]
head(data_class_final)
str(data_class_final) #making sure genre is a factor


dim(data_class_final)[1] #21569
21569/6 #3594 the ideal number of genre representation
1/6 #0.167
table(data_class_final$playlist_genre)/21569 #good representation for each genre, will take equal amounts from all

#75-25 split
.25*21569 #5392 #how many we need for a 25% test set
5392/6 #898.67 #how many we need from each genre to have a 25% test set


#doble checking we truly split the data set
dim(test1)
dim(train1)
5388+16181 #21569
dim(data_class_final) #21569 12 yay!

#formula in case we need it
class_formula2 <- paste("playlist_genre~",paste(selected_vars, collapse="+"), sep="")
class_formula2

#Doing 20 fold CV
set.seed(123)
k_folds <- createFolds(data_class_final$playlist_genre,k=20,list=F)
for (i in 1:20){
  print(table(data_class_final$playlist_genre[k_folds==i]))
}
sum(k_folds == 1)
1080/21569

B <- 20
all_error <- data.frame()
all_train_error <- data.frame()
set.seed(123)

for (b in 1:B){
  cat(b,"\n")
  row_error <- c()
  row_train_error <- c()
  
  train_cv <- data_class_final[k_folds!=b,]
  test_cv <- data_class_final[k_folds==b,]
  
  #KNN - k=1,3,5,7,9,11,13,15
  cat("KNN \n")
  for (kk in seq(1,15,by=2)){
    cat("k = ",kk," ")
    knn_cv_train_model <- knn(train_cv[,3:12], train_cv[,3:12], train_cv[,2], k=kk)
    knn_cv_model <- knn(train_cv[,3:12], test_cv[,3:12], train_cv[,2], k=kk)
    row_error <- c(row_error, mean(knn_cv_model != test_cv[,2]))
    row_train_error <- c(row_train_error, mean(knn_cv_train_model != train_cv[,2]))
  }
  cat("LDA ")
  #LDA - some variables are normal across the genres so this may work, covariance mx?
  model_cv_lda <- lda(train_cv[,3:12], train_cv[,2])
  row_error <- c(row_error, mean(predict(model_cv_lda, test_cv[,3:12])$class != test_cv$playlist_genre))
  row_train_error <- c(row_train_error, mean(predict(model_cv_lda, train_cv[,3:12])$class != train_cv$playlist_genre))
  
  
  #QDA - doing to account for non-normal variables
  cat("QDA ")
  model_cv_qda <- qda(train_cv[,3:12], train_cv[,2])
  row_error <- c(row_error, mean(predict(model_cv_qda, test_cv[,3:12])$class != test_cv$playlist_genre))
  row_train_error <- c(row_train_error, mean(predict(model_cv_qda, train_cv[,3:12])$class != train_cv$playlist_genre))
  
  
  #Naive Bayes - have independednt samples so should be accurate?
  cat("NB ")
  model_cv_nb <- naiveBayes(train_cv[,3:12], train_cv[,2])
  row_error <- c(row_error, mean(predict(model_cv_nb, test_cv[,3:12]) != test_cv$playlist_genre))
  row_train_error <- c(row_train_error, mean(predict(model_cv_nb, train_cv[,3:12]) != train_cv$playlist_genre))
  
  #Random Class. Forest
  cat("RF ")
  model_cv_rf  <- randomForest(as.formula(class_formula2), data=train_cv, ntree=700,mtry=2
                       importance=TRUE)
  row_error <- c(row_error, 1 - mean((predict(model_cv_rf, test_cv, type='class') == test_cv$playlist_genre)))
  row_train_error <- c(row_train_error, 1 - mean((predict(model_cv_rf, train_cv, type='class') == train_cv$playlist_genre)))
  
  #Multinom Logistic Regression
  cat("LR \n")
  model_cv_multinom <- multinom(as.formula(class_formula2), data=train_cv, trace=FALSE)
  row_error <- c(row_error, 1-mean((predict(model_cv_multinom, test_cv) == test_cv$playlist_genre)))
  row_train_error <- c(row_train_error, 1-mean((predict(model_cv_multinom, train_cv) == train_cv$playlist_genre)))
  
  all_error <- rbind(all_error, row_error)
  all_train_error <- rbind(all_train_error, row_train_error)
}

#column names for error df
error_df_names <- c("KNN1", "KNN3", "KNN5", "KNN7", "KNN9", "KNN11", "KNN13", "KNN15", "LDA", "QDA", "NB", "RF", "LR")

names(all_error) <- error_df_names
all_error_results <- data.frame(CV_Error=colMeans(all_error), CV_Var=apply(all_error, 2, var))
all_error_results[(which.min(all_error_results[,1])),]

write.csv(all_error, "class_cv_error.csv", row.names=FALSE) #errors for each fold

names(all_train_error) <- error_df_names
all_train_error_results <- data.frame(CV_Error=colMeans(all_train_error), CV_Var=apply(all_train_error, 2, var))
all_train_error_results[(which.min(all_train_error_results[,1])),]

write.csv(all_train_error, "class_cv_train_error.csv", row.names=FALSE) #mean and variance for errors

#RF, QDA, LR, LDA, NB, KNNs are all basically the same
all_error_results[order(all_error_results[,1]),]
all_train_error_results[order(all_train_error_results[,1]),]

#gets p val for every model against each other and making a plot
p_vals_all <- c()
dim(all_error)
for (i in 1:13){
  for (j in 1:13){
    p_vals_all <- c(p_vals_all, t.test(all_error[,i],all_error[,j],var.equal=TRUE,paired=TRUE)$p.value)
  }
}

reduced_p_vals_all <- matrix(data=p_vals_all, nrow=13, ncol=13, byrow=TRUE, dimnames=list(error_df_names, error_df_names))
reduced_p_vals_df_all <- data.frame(reduced_p_vals_all)
rpvdf_all <- reduced_p_vals_df_all %>% rownames_to_column() %>% gather(colname, value, -rowname)
#head(rpvdf_all)
rpvdf_all$tf <- rpvdf_all$value <= 0.05
#sum(rpvdf_all$tf)
summary(p_vals_all)
p_plot_all <- ggplot(data=rpvdf_all, aes(rowname, colname, fill=tf)) + xlab("Model 1") + ylab("Model 2") + ggtitle("T-Test Results: True mean of (Error of Model 1 - Error of Model 2) != 0") + geom_tile() + scale_fill_manual(values=c("red", "darkgreen"), guide=guide_legend(title="p<=0.05?")) 

p_plot_all

#Random Forest is the best one but it is not a good alg


#Build Final Random Forest Class Model

#going to do bagging
#doing resampling a bunch of times and then
rf_final  <- randomForest(as.formula(class_formula2), data=data_class_final, ntree=700, mtry=2, importance=TRUE)
1 - mean((predict(rf_final, data_class_final, type='class') == data_class_final$playlist_genre))
#0.000695
varImpPlot(rf_final)
#speechiness, danceability, tempo, energy are consistemtly important
# liveness and mode are consistently the bottom 2

table(predict(rf_final, data_class_final, type='class'))

rf_final_predict <- predict(rf_final, data_class_final, type='response')
length(rf_final_predict)
dim(data_class_final)

rf_cf <- confusionMatrix(predict(rf_final, data_class_final, type='class'), data_class_final$playlist_genre)
rf_cf$byClass[,1]
rf_cf$overall[1]
data_class_final[2732,]
#genre order: edm, latin, pop, r&b, rap, rock
#sensitivity: byClass col 1
#specificity: byClass col 2
#accuracy: overall element 1



#Bagging Alg Here
rf_sens <- c()
rf_spec <- c()
rf_acc <- c()

merge_test <- merge(data_class_final_preds, data.frame(track_id="3lTcHFrN025UJJwyAFYy1p", tester=5), all.x=TRUE, sort=FALSE)
head(merge_test)

B <- 100
data_class_final_preds <- data_class_final %>% dplyr::select(track_id, playlist_genre)

for (b in 1:B){
  
  cat(b, " ")
  test_bag <- slice_sample(data_class_final, n=898, by = "playlist_genre")
  train_bag <- data_class_final[!(data_class_final$track_id %in% test_bag$track_id),]
  
  rf_bag  <- randomForest(as.formula(class_formula2), data=train_bag, ntree=700, mtry=2, importance=TRUE)
  my_pred <- data.frame(track_id=test_bag[,1], pred=predict(rf_bag, test_bag, type='response'))
  hold <- merge(data_class_final_preds, my_pred, by="track_id", all=TRUE)
  data_class_final_preds <- cbind(data_class_final_preds, hold[,dim(hold)[2]])            
  
  # rf_bag_cf <- confusionMatrix(predict(rf_bag, test_bag, type='class'), test_bag$playlist_genre)
  # 
  # rf_acc <- c(rf_acc, rf_bag_cf$overall[1])
  # rf_sens <- rbind(rf_sens, rf_bag_cf$byClass[1])
  # rf_spec <- rbind(rf_spec, rf_bag_cf$byClass[2])
}

write.csv(data_class_final_preds, "genre_pred.csv", row.names=FALSE) #final predictions, not going to be used


head(data_class_final_preds)


getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

final_preds <- c()

#apply(data.frame(data_class_final_preds[1,][!is.na(data_class_final_preds[1,])]), 2, getmode)

for (i in 1:dim(data_class_final_preds)[1]){
  cat(i, " ")
  hold1 <- data.frame(data_class_final_preds[i,][!is.na(data_class_final_preds[i,])])

  final_preds <- c(final_preds, apply(hold1, 2, getmode))
}

#predicting whole data set on sampled data gives 0.8839 accuracy
sum(predict(rf_bag, data_class_final, type='response') == data_class_final$playlist_genre)



final_df <- data_class_final %>% dplyr::select(track_id, playlist_genre)

final_df <- cbind(final_df, final_preds)

#final predictions
write.csv(final_df, "final_genre_pred2.csv", row.names=FALSE) #final predictions for each song

sum(final_df$playlist_genre==final_df$final_preds)/21569
#0.1795169
#this model was plain old bad, maybe very underfit?

table(final_df$playlist_genre)
table(final_df$final_preds)

final_df[1,]
data_class_final_preds[1,][!is.na(data_class_final_preds[3,])]

#how many times the resulting prediction was voted for, isn't working correctly
num_votes <- c()

for (i in 1:dim(data_class_final_preds)[1]){
  cat(i, " ")
  #hold1 <- data.frame(data_class_final_preds[i,][!is.na(data_class_final_preds[i,])])
  
  num_votes <- c(num_votes, length(which(data_class_final_preds[i,c(-1,-2)]==final_preds[i])))
}


count_num_preds <- function(v){
  return (length(data.frame(v)[!is.na(data.frame(v))]))
}

num_preds <- c()
#how many times this row was chosed in the test set
for (i in 1:dim(data_class_final_preds)[1]){
  cat(i, " ")
  num_preds <- c(num_preds, count_num_preds(data_class_final_preds[i,c(-1,-2)]))
}

#final data frame for investigations
final_final_df <- cbind(final_df,num_votes, num_preds)

write.csv(final_final_df, "final_final_genre_pred2.csv", row.names=FALSE)



######################################
# Random Forest Cross Validation
######################################


#CV for RF parameters
tree_B <- seq(400,700,100)
mtry_B <- seq(1,dim(data_class_final)[2]-2)
nodesize_B <- seq(1,10)
#

rf_df <- c()
set.seed(123)
for (m1 in tree_B){
  cat(m1, " ")
  errors <- c()
  test_cv <- slice_sample(data_class_final, n=1080, by = "playlist_genre")
  train_cv <- data_class_final[!(data_class_final$track_id %in% test_cv$track_id),]


  for (m2 in mtry_B){
    cat(m2, " ")
      rf_cv <- randomForest(as.formula(class_formula2), data=train_cv, mtry=m2, importance=TRUE)
      #row <- c(row, 1 - mean((predict(rf_cv, auto_cv_test, type='class') == auto_cv_test$mpg01)^2))
      errors <- c(errors, 1 - mean((predict(rf_cv, test_cv, type='class') == test_cv$playlist_genre)^2))
    }



  rf_df <- rbind(rf_df, errors)
  cat("\n")
}

# find_rf_params <- function(x){
#   return (which(x==min(x),arr.ind=TRUE))
# }

dim(rf_df)
which.min(rf_df[4,])

rf_df[,which.min(rf_df)]
min(min_values)
# #going to say, n.tree=700, mtry=2, nodesize=1 aka same as default


#############################################3
# SINGLE RUN FOR ALL MODELS
#######################################3

test_error <- c()
train_error <- c()
set.seed(123)


test_error <- c()
train_error <- c()
  
test_cv <- slice_sample(data_class_final, n=1072, by = "playlist_genre") #30
train_cv <- data_class_final[!(data_class_final$track_id %in% test_cv$track_id),] #70
table(test_cv$playlist_genre)
table(train_cv$playlist_genre)
table(data_class_final$playlist_genre)
  
#KNN - k=1,3,5,7,9,11,13,15
cat("KNN \n")
for (kk in seq(1,15,by=2)){
  cat("k = ",kk," ")
  knn_cv_train_model <- knn(train_cv[,3:12], train_cv[,3:12], train_cv[,2], k=kk)
  knn_cv_model <- knn(train_cv[,3:12], test_cv[,3:12], train_cv[,2], k=kk)
  test_error <- c(test_error, mean(knn_cv_model != test_cv[,2]))
  train_error <- c(train_error, mean(knn_cv_train_model != train_cv[,2]))
}
cat("LDA ")
#LDA - some variables are normal across the genres so this may work, covariance mx?
model_cv_lda <- lda(train_cv[,3:12], train_cv[,2],alpha=.01)
test_error <- c(test_error, mean(predict(model_cv_lda, test_cv[,3:12])$class != test_cv$playlist_genre))
train_error <- c(train_error, mean(predict(model_cv_lda, train_cv[,3:12])$class != train_cv$playlist_genre))


#QDA - doing to account for non-normal variables
cat("QDA ")
model_cv_qda <- qda(train_cv[,3:12], train_cv[,2])
test_error <- c(test_error, mean(predict(model_cv_qda, test_cv[,3:12])$class != test_cv$playlist_genre))
train_error <- c(train_error, mean(predict(model_cv_qda, train_cv[,3:12])$class != train_cv$playlist_genre))


#Naive Bayes - have independednt samples so should be accurate?
cat("NB ")
model_cv_nb <- naiveBayes(train_cv[,3:12], train_cv[,2])
test_error <- c(test_error, mean(predict(model_cv_nb, test_cv[,3:12]) != test_cv$playlist_genre))
train_error <- c(train_error, mean(predict(model_cv_nb, train_cv[,3:12]) != train_cv$playlist_genre))

#Random Class. Forest
cat("RF ")
model_cv_rf  <- randomForest(as.formula(class_formula2), data=train_cv, 
                             importance=TRUE)
test_error <- c(test_error, 1 - mean((predict(model_cv_rf, test_cv, type='class') == test_cv$playlist_genre)))
train_error <- c(train_error, 1 - mean((predict(model_cv_rf, train_cv, type='class') == train_cv$playlist_genre)))

#Multinom Logistic Regression
cat("LR \n")
model_cv_multinom <- multinom(as.formula(class_formula2), data=train_cv, trace=FALSE)
test_error <- c(test_error, 1-mean((predict(model_cv_multinom, test_cv) == test_cv$playlist_genre)))
train_error <- c(train_error, 1-mean((predict(model_cv_multinom, train_cv) == train_cv$playlist_genre)))

single_run_error <- data.frame()
single_run_error <- rbind(single_run_error, train_error, test_error)
names(single_run_error) <- error_df_names

write.csv(single_run_error, "single_run_error.csv", row.names=FALSE)

table(data_class_final$playlist_genre)
table(train_cv$playlist_genre)
table(test_cv$playlist_genre)
rf_predictions <- predict(model_cv_rf, test_cv, type='class')

