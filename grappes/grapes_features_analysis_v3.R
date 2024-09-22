library(ggplot2)
library(dplyr)
library(SCBmeanfd)
library(data.table)
library(mgcv)
library(gratia)

#génère valeurs d'une normale multivariée
rmvn <- function(n, mu, sig) { 
  L <- mroot(sig)
  m <- ncol(L)
  t(mu + L %*% matrix(rnorm(m*n), m, n))
}



out_folder="C://Users//Utilisateur//Documents//IMS//VINELAPSE//GRAPPES//out//"
#features classiques estimées sur une fenêtre 32x32 ou 48x48 ou 64x64
df = read.csv('C://Users//Utilisateur//Documents//IMS//VINELAPSE//GRAPPES//features//feat_vinelapse.csv')
df <- df[,!names(df) %in% c("i_global", "i_patch")]
#features rayon des baies
df_rayon = read.csv('C://Users//Utilisateur//Documents//IMS//VINELAPSE//GRAPPES//features//feat_vinelapse_rayon.csv')
#Fusionner les deux, les rayons des baies seront dupliqués pour chaque patch_size (pas très joli mais flemme)
df_rayon$patch_size=32
df_rayon_sum=df_rayon
df_rayon$patch_size=48
df_rayon_sum=rbind(df_rayon_sum,df_rayon)
df_rayon$patch_size=64
df_rayon_sum=rbind(df_rayon_sum,df_rayon)

df=rbind(df,df_rayon_sum)


df$label=as.factor(df$label)
df$folder=as.factor(df$folder)
df$patch_size=as.factor(df$patch_size)
df$image=as.factor(df$image)
#calcul de la date depuis image
df$date=as.Date(substr(df$imag,6,9),format="%m%d")


#Remove banned features
liste_ban=c("GLDS_Contrast","GLDS_Mean")
#df=df[!(df$feature %in% liste_ban),]


#moyenne/std par image
df.summary <- df %>%
  group_by(folder,date,patch_size,label) %>%
  summarise(
    sd = sd(feature, na.rm = TRUE),
    mean = mean(feature),
    q5=quantile(feature, probs = 0.05, na.rm=TRUE),
    q25=quantile(feature, probs = 0.25, na.rm=TRUE),
    q50=quantile(feature, probs = 0.5, na.rm=TRUE),
    q75=quantile(feature, probs = 0.75, na.rm=TRUE),
    q95=quantile(feature, probs = 0.95, na.rm=TRUE)
  )
df.summary
events=c("Floraison","Nouaison","Petit pois","Fermeture", "Debut veraison", "Fin veraison")
events_date=as.Date(c("2024-05-30","2024-06-09","2024-06-25","2024-07-12","2024-08-05","2024-08-20"),format="%Y-%m-%d")
liste_features=unique(df$label)
liste_folder=unique(df$folder)
liste_patch_size=unique(df$patch_size)
#Visualisation basique avec moyenne et enveloppes
#======================
for (feature in liste_features) {
    ##Plot le summary
      ggplot(df.summary[df.summary$label==feature,])+
        geom_ribbon(aes(x=date, ymin=q5, ymax=q95),fill="gray30", alpha=0.2) +
        geom_ribbon(aes(x=date, ymin=q25, ymax=q75),fill="gray30", alpha=0.2) +
        geom_line(aes(x=date,y=mean),color="red") +
        geom_vline(xintercept=events_date, linetype="dotted",linewidth = 1)+
        theme_bw()+
        facet_grid(patch_size ~ folder) +
        xlab("Time")+
        ylab("Feature value") +
        labs(
          title = feature,
        )
      ggsave(paste0(out_folder,feature,"_qplot.png"), width = 6, height = 4)
}

#Test GAM
#==============
liste_resolution=c(1,7,14)
patch_size=48
mean_flag=TRUE
k=7
for (feature in liste_features) {
  if (!(feature %in% liste_ban)) {
    df_visu_sigslope=data.frame()
    df_visu_ribbon=data.frame()
    df_visu_feature=data.frame()
    for (folder in liste_folder) {
      for (resolution in liste_resolution) {
        if (mean_flag) {
          df_feature=df.summary[df.summary$label==feature & df.summary$patch_size==patch_size & df.summary$folder==folder,]
          names(df_feature)[names(df_feature) == "mean"] <- "feature"
        } else {
          df_feature=df[df$label==feature & df$patch_size==patch_size & df$folder==folder,]
        }
        if (!all(df_feature$feature == 0)){
          df_feature$x=as.integer(df_feature$date)-min(as.integer(df_feature$date))+1
          sampled_x=seq(1, max(df_feature$x), by=resolution)
          sampled_dates=seq(min(df_feature$date), max(df_feature$date), by=resolution)
          df_feature=df_feature[df_feature$x %in% sampled_x,]
          #Modèle GAM
          gam_y <- gam(feature~s(x,k=k),data=df_feature, method = "REML")
          #gam_y <- gam(feature~s(x),data=df_feature, method = "REML")
          #Premier intervale de confiance
          #Si on échantillone de façon répétée dans la population, 95% des intervales crées contiendront la véritable fonction
          #under repeated resampling from the population 95% of such confidence intervals will contain the true function
          #plot(gam_y, shade = TRUE, seWithMean = TRUE, residuals = TRUE, pch = 16, cex = 0.8)
          #mais cela ne décrit pas en réalité l'incertitude sur la fonction. si on tirait des splines depuis la posterior, la plupart ne seraient pas dans l'enveloppe
          #On cherche à générer un intervale simultané (approche de ruppert et al. (2003))
          Vb <- vcov(gam_y) #covariance des coeffs
          newd <- with(df_feature, data.frame(x = seq(min(x), max(x), length = 200))) #les x
          newd_date <- with(df_feature, data.frame(x_date = seq(min(date), max(date), length = 200))) #les x
          pred <- predict(gam_y, newd, se.fit = TRUE) #prédiction gam sur les x définis
          se.fit <- pred$se.fit #on extrait les écarts type des valeurs fittées
          set.seed(42)
          N <- 10000 #nombre de simus
          BUdiff <- rmvn(N, mu = rep(0, nrow(Vb)), sig = Vb) #0 car vecteur moyenne nul, avec cov vb
          Cg <- predict(gam_y, newd, type = "lpmatrix")
          simDev <- Cg %*% t(BUdiff) #différence modele simu
          absDev <- abs(sweep(simDev, 1, se.fit, FUN = "/"))
          masd <- apply(absDev, 2L, max)
          crit <- quantile(masd, prob = 0.95, type = 8) #valeur critique pour 95% ci
          pred <- transform(cbind(data.frame(pred), newd,newd_date),
                            uprP = fit + (2 * se.fit),
                            lwrP = fit - (2 * se.fit),
                            uprS = fit + (crit * se.fit),
                            lwrS = fit - (crit * se.fit)) #intevales de confiance point par point et simultané
          
          #on s'intéresse maintenant à la dérivée (la même avec le package gratia qui inclut tout)
          UNCONDITIONAL <- FALSE 
          N <- 10000             
          n <- 500               
          EPS <- 1e-07           
          fd <- fderiv(gam_y, newdata = newd, eps = EPS, unconditional = UNCONDITIONAL)
          set.seed(42)                            # set the seed to make this repeatable 
          sint <- confint(fd, type = "simultaneous", nsim = N)
          head(sint)
          #Find periods where the enveloppe is below or above 0
          sigslope=newd$x[which(sign(sint$lower)==sign(sint$upper))]
          sigslope_date=newd_date$x_date[which(sign(sint$lower)==sign(sint$upper))]
          ggplot(cbind(sint, x = newd$x,x_date=newd_date$x_date),
                 aes(x = x_date, y = est)) +
            geom_hline(aes(yintercept = 0)) +
            geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2) +
            geom_line() +
            geom_point(data=data.frame(x=sigslope_date,y=rep(0,length(sigslope))),aes(x=x,y=y),color="red")+
            labs(y = "First derivative", x = "Time") + theme_bw()
          ggsave(paste0(out_folder,"detail//",feature,resolution,folder,"_gamderiv_ci.png"), width = 6, height = 4)
          
          ggplot() +
            geom_point(data=df_feature, aes(x=date,y=feature),size=1,alpha=1,color="black") +
            geom_ribbon(data=pred,aes(x = x_date,ymin = lwrS, ymax = uprS), alpha = 0.1, fill = "red") +
            geom_ribbon(data=pred,aes(x = x_date,ymin = lwrP, ymax = uprP), alpha = 0.1, fill = "blue") +
            labs(y = "Feature",
                 x = "Time")+
            geom_point(data=data.frame(x=sigslope_date,y=rep(0,length(sigslope))),aes(x=x,y=y),color="red")+
            theme_bw()
          ggsave(paste0(out_folder,"detail//",feature,resolution,folder,"_gam_ci.png"), width = 6, height = 4)
          
          #Alimenter un data frame résumé pour toutes les images et les 3 résolutions
          df_visu_sigslope=rbind(df_visu_sigslope,cbind(sint, x = newd$x,x_date=newd_date$x_date,folder=folder, resolution=resolution))
          pred$folder=folder
          pred$resolution=resolution
          df_visu_ribbon=rbind(df_visu_ribbon,pred)
          df_feature$resolution=resolution
          df_visu_feature=rbind(df_visu_feature,df_feature)
          
          #plot(b,pages=1,residuals=TRUE)  ## show partial residuals
          #plot(b,pages=1,seWithMean=TRUE) ## `with intercept' CIs
          #par(mfrow = c(2, 2))
          #gam.check(gam_y)
          
          #gam_y |>
          #  basis() |>
          #  draw()
          #coef(gam_y) 
        }
      }
    }
    df_visu_ribbon$folder=as.factor(df_visu_ribbon$folder)
    df_visu_feature$folder=as.factor(df_visu_feature$folder)
    df_visu_ribbon$resolution=as.factor(df_visu_ribbon$resolution)
    df_visu_feature$resolution=as.factor(df_visu_feature$resolution)
    #visualiser pour cette feature
    ggplot(df_visu_sigslope,
           aes(x = x_date, y = est)) +
      geom_hline(aes(yintercept = 0)) +
      geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2) +
      geom_line() +
      facet_grid(resolution~folder) +
      #geom_point(data=data.frame(x=sigslope_date,y=rep(0,length(sigslope))),aes(x=x,y=y),color="red")+
      labs(y = "First derivative", x = "Time") + theme_bw()
    ggsave(paste0(out_folder,feature,resolution,folder,"_gamderiv_global.png"), width = 6, height = 4)
    
    ggplot() +
      geom_point(data=df_visu_feature, aes(x=date,y=feature),size=1,alpha=1,color="black") +
      geom_ribbon(data=df_visu_ribbon,aes(x = x_date,ymin = lwrS, ymax = uprS), alpha = 0.1, fill = "red") +
      geom_ribbon(data=df_visu_ribbon,aes(x = x_date,ymin = lwrP, ymax = uprP), alpha = 0.1, fill = "blue") +
      facet_grid(resolution ~folder ) +
      labs(y = "Feature",
           x = "Time")+
      #geom_point(data=data.frame(x=sigslope_date,y=rep(0,length(sigslope))),aes(x=x,y=y),color="red")+
      theme_bw()+ggtitle(feature)
    ggsave(paste0(out_folder,feature,"_gam_global.png"), width = 6, height = 4)
  }
}

