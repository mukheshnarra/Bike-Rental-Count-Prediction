#removing the global enviornment variables
rm(list=ls())

#setting working directory
setwd('D:/Edwisor/Bike Rental Count')

#loading required libraries
pacman::p_load(ggplot2,corrgram, DMwR,caret,randomForest,unbalanced,dummies,Information,MASS,rpart,gbm,ROSE,sampling,DataCombine, inTrees,fastDummies,rattle)

#reading data from csv files
bike_data=read.csv('day.csv',na.strings = c('NA'))

#printing the sample data
head(bike_data,5)

#knowing the structure of bike_data
str(bike_data)

#Expolartory analysis

#dropping variables is instant because which does'nt add any information
bike_data=subset(bike_data,select=-c(instant))

#Univariate Analysis
par(mfrow=c(4,3))
par(mar=rep(3,4))
hist(bike_data$season)
hist(bike_data$yr)
hist(bike_data$mnth)
hist(bike_data$holiday)
hist(bike_data$weekday)
hist(bike_data$workingday)
hist(bike_data$weathersit)
hist(bike_data$temp)
hist(bike_data$atemp)
hist(bike_data$hum)
hist(bike_data$windspeed)
hist(bike_data$casual)
hist(bike_data$registered)
hist(bike_data$cnt)
#As exploration we can find there are three dependent variable casual,registered,count
#where by summing up casual and registered gives count
#As expected, mostly working days and variable holiday and weekday is also showing a similar inference. You can use the code above to look at the distribution in detail.
#Variables temp, atemp, humidity and windspeed  looks naturally distributed. which are normalised

#converting variable into proper data type
bike_data$season=as.factor(bike_data$season)
bike_data$yr=as.factor(bike_data$yr)
bike_data$mnth=as.factor(bike_data$mnth)
bike_data$holiday=as.factor(bike_data$holiday)
bike_data$weekday=as.factor(bike_data$weekday)
bike_data$workingday=as.factor(bike_data$workingday)
bike_data$weathersit=as.factor(bike_data$weathersit)
#extracting the  day from dteday column
bike_data$dteday=sapply(bike_data$dteday,function(x) {format(as.Date(x,format='%Y-%m-%d'),'%d')})
bike_data$dteday=as.factor(bike_data$dteday)
 
#Missing values Analysis
Missing_values=data.frame(sapply(bike_data,function(x) {sum(is.na(x))}))
#setting Missing value count in column 
colnames(Missing_values)=c('Missing value count')
#where we did'nt find any missing values

#here we can make some hypothesis testing from data
#where casual users increases when weekend
par(mfrow=c(2,2))
boxplot(bike_data$casual~bike_data$weathersit,xlab='Weather',ylab='Casual')
boxplot(bike_data$registered~bike_data$weathersit,xlab='Weather',ylab='Registered')
#at seeing bike users incresing over weekends
boxplot(bike_data$casual~bike_data$weekday,xlab='Weekday',ylab='Casual')
boxplot(bike_data$registered~bike_data$weekday,xlab='Weekday',ylab='Registered')

#outlier Analysis
numeric_index=sapply(bike_data,is.numeric)
numeric_data=bike_data[,numeric_index]
cnames=colnames(numeric_data)
  for (i in 1:length(cnames)){
    assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "casual"), data = subset(bike_data))+ 
             stat_boxplot(geom = "errorbar", width = 0.5) +
             geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                          outlier.size=1, notch=FALSE) +
             theme(legend.position="bottom")+
             labs(y=cnames[i],x="casual")+
             ggtitle(paste("Box plot of casual users for",cnames[i])))
    
  }
  gridExtra::grid.arrange(gn1,gn2,gn3,gn4)
  
  for (i in 1:length(cnames)){
    assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "registered"), data = subset(bike_data))+ 
             stat_boxplot(geom = "errorbar", width = 0.5) +
             geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                          outlier.size=1, notch=FALSE) +
             theme(legend.position="bottom")+
             labs(y=cnames[i],x="registered")+
             ggtitle(paste("Box plot of registered users for",cnames[i])))
    
  }
  gridExtra::grid.arrange(gn1,gn2,gn3,gn4)
#where we found outliers in hum and windspeed but we neglecting those because of those natural outiers.
  


#feature engineering
#we can create some feature like temp_regr,temp_casual
    
regr_DT_tree=rpart(registered~temp,data=bike_data)
fancyRpartPlot(regr_DT_tree)
bike_data$temp_regr[bike_data$temp<0.27]=1
bike_data$temp_regr[bike_data$temp>=0.51]=2
bike_data$temp_regr[bike_data$temp>=0.27 & bike_data$temp<0.43]=3
bike_data$temp_regr[bike_data$temp>=0.43 & bike_data$temp<0.51]=4
bike_data$temp_regr=as.factor(bike_data$temp_regr)

casual_DT_tree=rpart(casual~temp,data=bike_data)
fancyRpartPlot(casual_DT_tree)
bike_data$temp_casual[bike_data$temp<0.32]=1
bike_data$temp_casual[bike_data$temp>=0.51]=2
bike_data$temp_casual[bike_data$temp>=0.32 & bike_data$temp<0.42]=3
bike_data$temp_casual[bike_data$temp>=0.42 & bike_data$temp<0.51]=4
bike_data$temp_casual=as.factor(bike_data$temp_casual)

#Feature Selection
correlation_data=data.frame(bike_data$temp,bike_data$atemp,bike_data$hum,bike_data$windspeed,bike_data$cnt,bike_data$casual,bike_data$registered)
corrgram::corrgram(correlation_data,order=F,upper.panel = panel.pie,text.panel =panel.txt,main='Correlation Plot')
l=cor(correlation_data)
corrplot::corrplot(l)
#removing temp variable from bike_data because atemp and temp has highly correlated
#where priting the correlation between the variables
#as you find there is high correlation between the temp and atemp varibales
#where after creating varaible we are deleting the temp variable
bike_data=subset(bike_data,select=-c(temp))


#removing the all variable from global enviornment except the bike_data
rmExcept(c('bike_data'))

#sampling where we spliting the data into train and test split in 80:20 ratio
train_index=sample(1:nrow(bike_data),0.8*nrow(bike_data))
train_data=bike_data[train_index,]
test_data=bike_data[-train_index,]

#Modelling
#where here we have three  dependent variables
#we have to apply model two times to get the two dependent variables like casual and registered and total gives you the cnt
#where casual and registered is little skewed so we apply log transformation on both variables

--------------------------------------#Decision Tree--------------------------------------------
#model to predict casual variable
casual_DT=rpart(log(casual)~dteday+season+yr+mnth+holiday+weekday+workingday+temp_casual+weathersit+atemp+hum+windspeed,data=train_data,method='anova')
summary(casual_DT)
#visualize the decision tree for predicting the casual variable
fancyRpartPlot(casual_DT)
predictions_casual_DT=predict(casual_DT,test_data[,-c(12,13,14,15)])
predictions_casual_DT=exp(predictions_casual_DT)

#model to predict registered variable
registerd_DT=rpart(log(registered)~dteday+season+yr+mnth+holiday+temp_regr+weekday+workingday+weathersit+atemp+hum+windspeed,data=train_data,method='anova')
summary(registerd_DT)
#visualize the decision tree for predicting the registered
fancyRpartPlot(registerd_DT)
predictions_registered_DT=predict(registerd_DT,test_data[,-c(12,13,14,16)])
predictions_registered_DT=exp(predictions_registered_DT)

#sum of predictions of casual variable  and registered will be give predictions of total count
predictions_DT=predictions_casual_DT+predictions_registered_DT

-------------------------------#RandomForrest#--------------------------------------------------
#model to predict casual variable
casual_RF=randomForest(log(casual)~dteday+season+yr+mnth+holiday+temp_casual+weekday+workingday+weathersit+atemp+hum+windspeed,data=train_data)
predictions_casual_RF=predict(casual_RF,test_data[,-c(12,13,14,15)])
predictions_casual_RF=exp(predictions_casual_RF)

#model to predict registered variable
registerd_RF=randomForest(log(registered)~dteday+season+yr+mnth+holiday+temp_regr+weekday+workingday+weathersit+atemp+hum+windspeed,data=train_data)
predictions_registerd_RF=predict(registerd_RF,test_data[,-c(12,13,14,16)])
predictions_registerd_RF=exp(predictions_registerd_RF)

#sum of predictions of casual variable  and registered will be give predictions of total count
predictions_RF=predictions_casual_RF+predictions_registerd_RF

#we can visualize the rules of random forrest for casual  variable
treelist_casual=RF2List(casual_RF)
Rules_casual=extractRules(treelist_casual,train_data[,-c(12,13,14,15)])
readableRules_casual=presentRules(Rules_casual,colnames(train_data))

#we can visualize the rules of random forrest for casual  variable
treelist_registered=RF2List(registered_RF)
Rules_registered=extractRules(treelist_registered,train_data[,-c(12,13,14,16)])
readableRules_registered=presentRules(Rules_registered,colnames(train_data))
------------------------------#linear regression-------------------------------------------------
#model to predict casual variable
#where for categorical variable dummy variables are created
casual_lg=lm(log(casual)~dteday+season+yr+mnth+holiday+weekday+workingday+weathersit+temp_casual+atemp+hum+windspeed,data=train_data)
predictions_casual_lg=predict(casual_lg,test_data[,-c(12,13,14,15)])
predictions_casual_lg=exp(predictions_casual_lg)
#printing the summary of linear regression
summary(casual_lg)

#model to predict registered variable
registerd_lg=lm(log(registered)~dteday+season+yr+mnth+holiday+weekday+temp_regr+workingday+weathersit+atemp+hum+windspeed,data=train_data)
predictions_registerd_lg=predict(registerd_lg,test_data[,-c(12,13,14,16)])
predictions_registerd_lg=exp(predictions_registerd_lg)
#printing the summary of linear regression
summary(registerd_lg)
predictions_lg=predictions_casual_lg+predictions_registerd_lg

#function for evaluation
mape=function(y,y_pred)
{
  mean(abs(y-y_pred)/y)*100
}

#checking mean absolute persentage error 
#for random forrest
mape(test_data[,14],predictions_RF)#--->17.80648
#for linear regression
mape(test_data[,14],predictions_lg)#--->17.86032
#for decision tree
mape(test_data[,14],predictions_DT)#--->26.09761

#so we are choosing the random forrest algorithm because it less mape value
#----------------------------------#Hyperparameter Tuning------------------------------------
#so furthur moving we can fine tune the random forrest
#where ntree and mtry are varibales used for tuning random forrest
plot(casual_RF)
#as for casual variable we see from plot the at 200 trees the error became static
plot(registerd_RF)
#as for registered variable we see from plot the at 200 trees the error became static
x_casual<-train_data[,-c(12,13,14,16)]
y_casual<-train_data[,12]
bmtry<-tuneRF(x_casual,y_casual,ntreeTry = 200,stepFactor = 1.5, improve = 1e-5)
bmtry
#as we for mtry variables for random split is 3 where oob error is low

x_regr<-train_data[,-c(12,13,14,15)]
y_regr<-train_data[,13]
bmtry<-tuneRF(x_regr,y_regr,ntreeTry = 500,stepFactor = 1.5, improve = 1e-5)
bmtry
#as we for mtry variables for random split is 3 where oob error is low

#train again and check if accuarcy is increase or not


#model to predict casual variable
casual_RF=randomForest(log(casual)~dteday+season+yr+mnth+holiday+temp_casual+weekday+workingday+weathersit+atemp+hum+windspeed,data=train_data,mtry=3,ntree=200)
predictions_casual_RF=predict(casual_RF,test_data[,-c(12,13,14,15)])
predictions_casual_RF=exp(predictions_casual_RF)

#model to predict registered variable
registerd_RF=randomForest(log(registered)~dteday+season+yr+mnth+holiday+temp_regr+weekday+workingday+weathersit+atemp+hum+windspeed,data=train_data,mtry=3,ntree=200)
predictions_registerd_RF=predict(registerd_RF,test_data[,-c(12,13,14,16)])
predictions_registerd_RF=exp(predictions_registerd_RF)

#sum of predictions of casual variable  and registered will be give predictions of total count
predictions_RF=predictions_casual_RF+predictions_registerd_RF

mape(test_data[,14],predictions_RF)
#there is only minute change in accuracy

#creating new variable in test_data for casual prediction
test_data['casual_prediction_cnt']=predictions_casual_RF
#creating new variable in test_data for casual prediction
test_data['registered_prediction_cnt']=predictions_registerd_RF
#creating new variable in test_data for prediction
test_data['prediction_cnt']=predictions_RF

#writing the data into new csv file
write.csv(test_data,'bike_rental_prediction_R.csv',row.names = F)
