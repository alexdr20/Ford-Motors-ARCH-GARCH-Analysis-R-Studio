library(tidyverse)
library(fGarch)
library(forecast)
library(fDMA)
library(lmtest)
library(FinTS)
library(tseries)
library(readxl)
library(lubridate)
library(nortsTest)
library(urca)

#importul datelor
F <- read.csv("C:/Users/user/Downloads/F.csv")
View(F)
ford=F[,-c(2,3,4,6,7)] #din F.csv am pastrat doar coloanele cu data,respectiv cu preturile zilnice de inchidere 
View(ford)

#setul de date
serie=ford

#declararea seriei de timp
t <- ts(serie$Close, start = decimal_date(as.Date("2015-01-02")), frequency = 365)

# Graficul seriei cu preturile zilnice de inchidere 
autoplot(t) + theme_bw() + ylab('Close price') + 
  ggtitle('Close price pentru perioada ian 2015 - dec 2022') +
  theme(plot.title = element_text(hjust = 0.5))

#histograma probabilitati de distributie preturi zilnice ded inchidere
options(repr.plot.width=21, repr.plot.height=11)#to set the figure size
hist(ford$Close,prob=T,breaks=50,xlab="Daily Returns",main = "Valori Close Price Ford perioada ian 2015 - dec 2022",
     ylab="Probabilitatea de distribuție",col="cornflowerblue") 
mu<-mean(ford$Close)  
sigma<-sd(ford$Close)
x<-seq(min(ford$Close),max(ford$Close),length=80) 
y<-dnorm(x,mu,sigma) 
lines(x,y,lwd=2,col="red")

#calcul rentabilitate prin diferentiere 
t_returns <- diff(log(t))

# Graficul rentabilitatii 
autoplot(t_returns) + theme_bw() +
  ylab('Rentabilitate') + 
  ggtitle('Rentabilitate zilnica actiuni FORD perioada ian 2015- dec 2022') +
  theme(plot.title = element_text(hjust = 0.5)) 

#split pe seturi de testare si antrenare pentru seria rentabilitatilor
#consideram 70% din obs ca fiind in testul de antrenare (1410-pana la data 2020-08-07),iar restul de 30% (604 obs) ca fiind in setul de testare
ford.train=window(t_returns,start = decimal_date(as.Date("2015-01-02")),end=decimal_date(as.Date("2020-08-07")))
ford.train

ford.test=tail(t_returns,604)
ford.test

#-------------------ESTIMARE ARCH-----------
# Modelele ARCH au la baza doua ecuatii: a mediei si a dispersiei
# Ecuatia mediei:
# yt = theta0 + theta1*yt-1 + ... + thetan*yt-n + epsilont
# Ecuatia variantei:
# ht = alpha0 + alpha1*h t-1^2 + ... + alphan*h t-n^2
# Cu urmatoarele proprietati:
# toti alpha > 0 - pentru a garanta varianta pozitiva
# 0 <= SUM(alpha) <= 1 - toti alpha insumati trebuie sa fie intre 0 si 1
# alpha1 > alpha2 > ... > alphan trecutul recent trebuie sa aiba mai multe influenta decat trecutul vechi

# Pasii pe care trebuie sa ii urmam cand estimam ARCH 
# Partea 1: ecuatia mediei 
# Pas 1: Stationaritatea seriei (diferentiem daca e nevoie)
# Pas 2: Estimam ecuatia mediei (ARIMA) 
# Partea 2: ecuatia dispersiei 
# Pas 1: Verificam daca exista efecte ARCH (testul ARCH-LM)
# Pas 2: Estimam modelul ARCH

#1.1 verificare STATIONARITATE serie

#pentru seria initiala vom considera testul ADF+TREND
adf_trend <- ur.df(t, type='trend', selectlags = c("AIC"))
summary(adf_trend) # serie nestationara deoarece |test statistic| < |critical values|

#pentru seria diferentiata doar testul ADF (fara TREND)
adf_none <- ur.df(t_returns, type='none', selectlags = c("AIC")) # verificam pentru type = none deoarece nu mai are trend seria
summary(adf_none) # serie stationara |test statistic| > |valori critice|

#1.2 estimare ecuatia mediei folosind ARIMA pe baza setului de antrenare
ggtsdisplay(t_returns, lag.max = 36)

#conform graficelor ACF si PACF lag-urile de la care se poate compune modelul ARMA încep de la nivelul 5,acestea fiind primele valori care ies în afara intervalului.
#După mai multe încercări la nivel de calcule,am ales a estima un model AR(4) și MA(4) pentru a vedea semnificația statistică 
arima404 <- Arima(ford.train, order = c(4,0,4), include.constant = TRUE)
coeftest(arima404) # coeficienti semnificativi
summary(arima404)

#2.1 verificare prezenta efecte ARCH pentru 1,respectiv 2 laguri in cadrul modelului arima404
ArchTest(residuals(arima404), lag = 1)
ggPacf(residuals(arima404)^2, 12) #conform graficului am putea merge cu analiza pana la lagul 6 
ArchTest(residuals(arima404), lag = 2) # p < 0.1 => efecte ARCH 
ArchTest(residuals(arima404), lag = 6) #in continuare p < 0.1 => efecte ARCH

#2.2 estimare model ARCH pe baza setului de antrenare
#pentru a usura calculele putem alege un model de tipul ARCH(2,0)
arch.fit2 <- garchFit(~arma(4,4) + garch(2,0), data = ford.train, trace = T)
summary(arch.fit2)
#prezenta efecte ARCH p-value < 0.1
#nu exista autocorelatie in medie
#exista autocorelatie in varianta

#verificare si pentru model ARCH(6,0)
arch.fit6 <- garchFit(~arma(4,4) + garch(6,0), data = ford.train, trace = T)
summary(arch.fit6) 
#la lagul 6 deja prezenta efectelor de tip ARCH nu mai este prezenta 
#p-value corespunzator este mult superior valorii de referinta de 0.1
#nu exista autocorelare in medie si nici in varianta

# Verificare proprietate 2 - SUM(coef alfa) sa fie intre 0 si 1
0.4353+0.2163 # =0.6516 pt lag 2
0.3652+0.05911+0.1170+0.00000001+0.1700+0.05041 # =0.76172 pt lag 6
#cu cat suma acestor coeficienti este mai aproape de 1,volatilitatea este mai persistenta 

#grafic varianta conditionata 
plot(arch.fit2) #Conditional SD
plot(arch.fit6)


#-----------------------ESTIMARE GARCH----------
# Modelul GARCH este o extensie a modelului ARCH care permite o alterinativa 
# pentru estimarea modelelor ARCH cu ordin mare


# Modelele GARCH au la baza tot doua ecuatii: a mediei si a dispersiei
# Ecuatia mediei:
# yt = theta0 + theta1*yt-1 + ... + thetan*yt-n + epsilont
# unde epsilont | It-1 N(0,ht) - formeaza o distributie normala cu varianta hetero
# Ecuatia variantei:
# ht = alpha0 + SUMA alphai*epsilon t-1^2 + SUMA betai*ht-i
# SUMA alphai*epsilon t-1^2 => ARCH
# SUMA betai*ht-i => includerea termenilor MA in ecuatia variantei
# Cu urmatoarele proprietati:
# alpha0 > 0, alphai > 0, betai > 0 pentru a garanta varianta pozitiva
# 0 <= SUMA alphai + SUMA beta1 < 1 pentru a asigura descompunerea variantei


# Pasii pe care trebuie sa ii urmam cand estimam GARCH 
# Partea 1: ecuatia mediei 
# Pas 1: Stationaritatea seriei (diferentiem daca e nevoie)
# Pas 2: Estimam ecuatia mediei (ARIMA) 
# Partea 2: ecuatia dispersiei 
# Pas 1: Verificam daca exista efecte ARCH (testul ARCH-LM) si daca avem hetero la 
# laguri superioare ne ducem in modele de tip GARCH
# Pas 2: Estimam modelul GARCH(1,1)
# Pas 3: Diagnostic pe reziduuri
# Pas 4: Daca se pastreaza heteroschedasticitatea, respecificam modelul GARCH 

#la testul GARCH partea de noutate vine la partea de estimare a modelului
#urmand aceeasi pasi ca pentru testul ARCH,vom incepe analiza de la partea 2.2
#estimare model GARCH(1,1)

garch.fit <- garchFit(~arma(4,4) + garch(1,1), data = ford.train, trace = T)
summary(garch.fit)
#nu exista efecte ARCH
#nu exista autocorelatie in medie,nici in varianta

#verificare propietati model // SUM(alfa+beta) sa fie intre 0 si 1 
0.05124+0.9392 # = 0.99044
#in cadrul modelului GARCH(1,1) volatilitatea este mult mai bine explicata
#suma coeficientilor fiind cel mai aproape de 1 

#grafic varianta conditionata
dev.new()
plot(garch.fit)

#prognoza folosing GARCH(1,1) pe baza datelor din setul de testare
#cum setul de training se termina la luna august
#orizontul de previziune va incepe de la urmatoarea luna
#in mod evident,cum am dorit estimarea pentru 30 zile,luna previzionata va fi septembrie

library(rugarch)
dev.new()
ug_spec = ugarchspec(mean.model = list(armaOrder = c(4,4)))
ugfit = ugarchfit(spec = ug_spec, data = ford.test)
ugfore = ugarchforecast(ugfit, n.ahead = 30)
summary(ugfore)
plot(ugfore) 


#se poate realiza graficul si doar pentru perioada dorita a fi prognozata
#folosing GARCH(1,1) folosind intreaga serie a rentabilitatilor se poate prognoza urmatoarea perioada
#cum datele folosite sunt zilnice,am ales a prognoza un grafic pentru 30 zile,respectiv pentru luna ian 2023
#ulterior,se pot face comparatii,la nivel vizual(de grafic) cu valorile prognozate si cele reale inregistrate in luna respectiva

s1 <- ugarchspec(variance.model=list(model="sGARCH",garchOrder=c(1,1)),
                 mean.model=list(armaOrder=c(4,4)),distribution.model="norm")
m1 <- ugarchfit(data = t_returns, spec = s1)

dev.new()
plot(m1, which = 'all')


s1final <-  ugarchspec(variance.model=list(model="sGARCH",garchOrder=c(1,1)),
                       mean.model=list(armaOrder=c(4,4)),distribution.model="norm")
m1final <- ugarchfit(data = t_returns, spec = s1final)
f <- ugarchforecast(fitORspec = m1final, n.ahead = 30)
plot(fitted(f))
sfinal <- s1
setfixed(sfinal) <- as.list(coef(m1))



sim <- ugarchpath(spec = sfinal,
                  m.sim = 1,
                  n.sim = 1*30)
plot.zoo(fitted(sim))

pred_ian_2023<- apply(fitted(sim), 2, 'cumsum') 
matplot(pred_ian_2023, type = "l", lwd = 3)