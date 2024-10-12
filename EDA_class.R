# EDA for Spotify Classification

library(DataExplorer)
library(corrplot)
library(ggcorrplot)
library(tidyverse)
library(tools) #for title case
library(caret) #for confusion mx

getwd()
setwd("C:/Users/Kerstin/Desktop/GT/7406/Project")



data <- read.csv("spotify_clean_top90.csv", sep=",", header=TRUE)
pred <- read.csv("final_genre_pred2.csv", sep=",", header=TRUE)

full_data <- read.csv("spotify_clean.csv", sep=",", header=TRUE)
full_pred <- read.csv("genre_pred.csv", sep=",", header=TRUE)

head(data)
head(pred)
head(full_data)
head(full_pred)
#######################################
# RAW DATA
#######################################
names(data)
data2 <- data[,c(1,4,10,12:23)]
head(data2)

unique(data2$playlist_genre)

pop_data <- data2 %>% filter(playlist_genre == "pop")
rock_data <- data2 %>% filter(playlist_genre == "rock")
rap_data <- data2 %>% filter(playlist_genre == "rap")
latin_data <- data2 %>% filter(playlist_genre == "latin")
rb_data <- data2 %>% filter(playlist_genre == "r&b")
edm_data <- data2 %>% filter(playlist_genre == "edm")


# BOXPLOTS
plot_boxplot(data2,by="playlist_genre")

#CORRELATION
corrplot(cor(data2[,c(-1,-3)]), method="color")

cor_labs <- c("track_popularity", "edm", "latin", "pop", "r&b", "rap", "rock", names(data2)[4:15])

data3 <-model.matrix(~0+., data=data2[,-1])
head(data3)
biserial.cor(data3[,"danceability"],data3[,7], level=2) #level=2 means the value 1 is the reference level

model.matrix(~0+., data=data2[,-1]) %>% 
  cor(use="pairwise.complete.obs", method="spearman") %>% 
  ggcorrplot(show.diag=FALSE, legend.title = "Correlation") + scale_x_discrete(labels = cor_labs) + scale_y_discrete(labels=cor_labs)


#HISTOGRAMS
names(data2)

#want to investigate danceability, energy, loudness, speechiness, acoutsicness, instrumentalness, valence, duration_ms
#they have the strongest correlations with the genres
hist_vars <- c("danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "valence", "duration_ms")
length(hist_vars) #8
unique(pred$playlist_genre) #pop, rap, rock, latin, rb, edm

par(mfrow=c(2,4))
for (i in 1:8){
  hist(unlist(rap_data %>% select(hist_vars[i])), main=paste("Rap ", toTitleCase(hist_vars[i])), xlab="")
}
dev.off()


#######################################
# PREDICTION DATA
#######################################

#number of times each row was predited
summary(apply(!is.na(full_pred[c(-1,-2),]), 1, sum))

head(pred)

mean(pred$playlist_genre != pred$final_preds)
#0.8203 classified incorrectly

#num rows
dim(pred)[1] #21569
21569/6 #3594.833
1/6 #0.167

table(pred$playlist_genre)
table(pred$playlist_genre)/21569

table(pred$final_preds)
table(pred$final_preds)/21569

#we got very similaor proportion of predictions for each genre

#confusion matri to look at misclassification
conf_mx <- confusionMatrix(data=as.factor(pred$final_preds), reference = as.factor(pred$playlist_genre))

#diagonal is terrible haha, looks like rap was predicted the best
#each genre has the most misclassifiations as rap, rap was called pop the most
#mcnemar's test shows that there is a significant difference in each class' proportion of errors
    # model made errors for each class in different proportions
#kappa = 0.0081 means very low agreement between predition and actual, duh
#sensitivity = correct positive/total positive, very low for all classes
# specificity = correct negative/total negative, very high across classes
# prevalence = actual class/all, balanced around 16% for all classes
#pos pred value and neg pred value are analogous to above
# detection rate = TP/all, less than 6% across all classes
# detection prevalence = TP+TN/all, 9%-26%
# balanced accuracy = spec+sens/2, around 50% for all classes
plot_confusion_matrix(conf_mx)

heatmap(table(as.factor(pred$playlist_genre), as.factor(pred$final_preds)),main = "Confusion Matrix",
        xlab = "Predicted",
        ylab = "Actual",
        col = heat.colors(10),
        scale = "column")

dev.off()

#artist representation and popularity
artist_pop <- full_data %>% dplyr::select(track_artist, track_popularity) %>%
  group_by(track_artist) %>%
  summarise(n=n(), w_avg=sum(track_popularity)*(n/21569)) %>%
  arrange(desc(w_avg))

artist_pop <- data.frame(artist_pop)
head(artist_pop, 50)
summary(artist_pop) #10,026 artists total

hist(head(artist_pop$n, 100))
artist_pop %>% arrange(desc(n))

# should I match songs with 

head(pred)
song_info <- merge(pred, full_data %>% select(track_id, track_name, track_artist), by="track_id")[,-1]
dim(song_info) #21569 4
song_info %>% filter(track_artist == "Taylor Swift")
song_info %>% filter(track_artist == "Daddy Yankee")
song_info %>% filter(track_artist == "Metallica")
song_info %>% filter(track_artist == "The Chainsmokers")
table(song_info %>% filter(track_artist == "Queen") %>% select(final_preds))
table(song_info %>% filter(track_artist == "Queen") %>% select(playlist_genre))

#Studying one artist/track
artist_pop %>% filter(track_artist == "Rihanna") #27 tracks, tied ranked for 29th, pop=1.811 ranked 41
song_info %>% filter(track_artist == "Rihanna") #this data emphasizes that these are PLAYLIST genres, not song genres
full_data %>% filter(track_name=="Don't Stop The Music")
full_data %>% filter(playlist_id=="6a66cg3HcsjYkisYyQcov6")

full_data %>% filter(track_artist == "The Weeknd")
full_data %>% filter(playlist_subgenre == "trap")




# FOR AMEN
genres <- unique(pred$playlist_genre)
par(mfrow=c(1,6))
hist(unlist(rock_data$track_popularity), main="Rock Popularity")



