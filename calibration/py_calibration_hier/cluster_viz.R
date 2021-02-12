library(RSQLite)
con.output <- dbConnect(RSQLite::SQLite(), "~/git/impala/calibration/py_calibration_hier/results/Ti64/res_ti64_clst4.db")

con.input <- dbConnect(RSQLite::SQLite(), "~/git/impala/calibration/py_calibration_hier/data/data_Ti64.db")

dbListTables(con.output)
clusters<-dbReadTable(con.output, "delta")

dat<-dbReadTable(con.input, "meta")

all(dat$table_name == dbReadTable(con.output, "meta")$table_name)


phis<-dbReadTable(con.output, "phis")
phisi<-phis[phis$cluster==1,]
pairs(phisi[,-c(1:2)],cex=.1)

ind<-matrix(nrow=2001,ncol=ncol(clusters))
for(i in 1:2001){
  for(j in 1:ncol(clusters)){
    cl<-clusters[i,j]
    ind[i,j]<-which(phis$iteration==(i-1) & phis$cluster==cl)
  }
}

pairs(phis[ind[,30],-c(1:2)],cex=.2)
pairs(phis[ind[,2],-c(1:2)],cex=.2)

png('~/Desktop/cluster_parameters.png',width=20,height=20,units="cm",res=300, pointsize=6)
pairs(phis[c(ind[,c(1:3)]),-c(1:2)],cex=.2,col=rep(1:3,each=2001),lower.panel=NULL)
legend(0,.5,dat$fname[1:3],col=1:3,pch=1,xpd=T,cex=1)
dev.off()

png('~/Desktop/cluster_parameters2.png',width=20,height=20,units="cm",res=300, pointsize=6)
pairs(phis[c(ind[,c(69,85,111)]),-c(1:2)],cex=.2,col=rep(1:3,each=2001),lower.panel=NULL)
legend(0,.5,dat$fname[c(69,85,111)],col=1:3,pch=1,xpd=T,cex=1)
dev.off()

jaa<-matrix(nrow=197,ncol=197)
for(i in 1:197){
  for(j in 1:197){
    jaa[i,j]<-jaa[j,i]<-sum(clusters[,i]==clusters[,j])
  }
}

jaa[1:15,1:15]

jaa[,c(1,14)]


clsts<-list()
j<-1
jaa2<-jaa
while(sum(jaa2)>0){
  clsts[[j]]<-which(jaa2[,j]>100)
  jaa2[clsts[[j]],]<-jaa2[,clsts[[j]]]<-0
  j<-j+1
}
clsts[sapply(clsts,length)==0]<-NULL

length(clsts)
length(unlist(clsts))
clsts[[3]]

png('~/Desktop/cluster_edot.png',width=10,height=10,units="cm",res=300, pointsize=6)
ord<-order(dat$edot)
fields::image.plot(jaa[ord,ord]/2001,col = rev(heat.colors(100)),main='Experiment clustering ordered by edot',xaxt='n',yaxt='n')
dev.off()

png('~/Desktop/cluster_temp.png',width=10,height=10,units="cm",res=300, pointsize=6)
ord<-order(dat$temperature)
fields::image.plot(jaa[ord,ord]/2001,col = rev(heat.colors(100)),main='Experiment clustering ordered by temp',xaxt='n',yaxt='n')
dev.off()

png('~/Desktop/cluster_paper.png',width=10,height=10,units="cm",res=300, pointsize=6)
ord<-order(dat$pname)
fields::image.plot(jaa[ord,ord]/2001,col = rev(heat.colors(100)),main='Experiment clustering ordered by paper',xaxt='n',yaxt='n')
dev.off()

png('~/Desktop/cluster_unknown.png',width=10,height=10,units="cm",res=300, pointsize=6)
ord<-unlist(clsts)
fields::image.plot(jaa[ord,ord]/2001,col = rev(heat.colors(100)),main='Experiment clustering ordered by ??',xaxt='n',yaxt='n')
dev.off()


ord<-order(log(dat$edot*1e6)*dat$temperature)
fields::image.plot(jaa[ord,ord]/2001,col = rev(heat.colors(100)),main='Experiment clustering ordered by log(edot)*T',xaxt='n',yaxt='n')




dat$fname[c(1:3,40)]

image(jaa[which(dat$temperature>1000),which(dat$temperature>1000)])

image(jaa[which(dat$temperature<400),which(dat$temperature<400)])




# aa<-read.csv("~/git/impala/calibration/py_calibration_hier/deltati64.csv")[,-1]
# dat<-read.csv('~/git/impala/calibration/py_calibration_hier/data/datti64.csv')
# dat2<-read.csv('~/git/immpala/code/visualization_tool/tmp/my_data.csv')
# pap2<-do.call(rbind,strsplit(do.call(rbind,strsplit(dat2$FILE_Visar_1,'/'))[,3],'[.]'))[,1]
# o1<-order(pap2)
# dat2<-dat2[o1,]
# o<-order(dat$X5)
# dat<-dat[o,]
# aa<-aa[,o]

ind<-which(heat.mat[,2]>.8)
dat2[ind,]

nsub = nrow(dat)
heat.mat<-matrix(0,nrow=nsub,ncol=nsub)
for(i in 1:nsub){
  for(j in 1:nsub){
    heat.mat[i,j]<-mean(aa[,i]==aa[,j])
  }
}
image(1:nsub,1:nsub,1-heat.mat,col = heat.colors(100))

ord<-order(dat$edot)
image(1:nsub,1:nsub,1-heat.mat[ord,ord],col = heat.colors(100))

cl.list<-list()
for(i in 1:48)
  cl.list[[i]]<-which(aa[2001,]==i)

plot(dat$edot,dat$temperature,col=as.numeric(aa[2001,]))

plot(dat$edot,dat$temperature,pch=as.character(aa[2001,]))




library(randomForest)
mod<-randomForest(x=dat[1:196,c(2,3,5)],y=as.factor(aa[100,1:196]))
mod$confusion[,ncol(mod$confusion)]

# cluster membership cannot be predicted with temp, edot, paper



m<-1000
cols<-colors()[seq(1,length(colors()),3)][as.numeric(aa[m,])]
pch<-rep(1:10,5)[as.numeric(aa[m,])]
plot(jitter(dat$edot,50),jitter(dat$temperature,50),col=cols,pch=pch)


swap.sym<-function(mat,i,j){#swap spots for a symmetric matrix (think of corr matrix, preserves corr)
  perm<-diag(ncol(mat))
  perm[i,i]<-perm[j,j]<-0
  perm[i,j]<-perm[j,i]<-1
  return(perm%*%mat%*%t(perm))
}


sub.clust<-1:nsub
heat.mat.clust<-heat.mat
for(j in 1:(nsub-1)){
  k=0
  for(i in j:(nsub-1)){
    k=k+1
    ord<-order(heat.mat.clust[(j+1):nsub,j],decreasing=T)+j
    temp<-sub.clust[i+1]
    sub.clust[i+1]<-sub.clust[ord[k]]
    sub.clust[ord[k]]<-temp
    heat.mat.clust<-swap.sym(heat.mat.clust,ord[k],i+1)
  }
}
par(mar=c(5,3,3,5),cex=.9)
library(fields)
image.plot(1:nsub,1:nsub,heat.mat.clust,col=rev(heat.colors(100)),xaxt='n',yaxt='n',ylab='',xlab='subject')
axis(1,at=1:nsub,labels=sub.clust)
image.plot(heat.mat,col=rev(heat.colors(100)),legend.only = T)







heat.mat<-heat.mat[o,o]

ord<-order(dat$X2)
image(1:nsub,1:nsub,1-heat.mat[ord,ord],col = heat.colors(100))

which.max(colSums(heat.mat))
ind<-which(heat.mat[,48]>.8)
dat2[ind,]

plot(dat2$Temperature,dat2$Strain_rate)
points(dat2$Temperature[ind],dat2$Strain_rate[ind],col=2)

xx<-dat2[,2:14]
xx<-apply(xx,2,scale)
ss<-svd(xx)
plot(ss$d^2)
ord<-order(ss$u[,12])
image(1-heat.mat[ord,ord],col = heat.colors(100))

ord<-order(xx[,2]*xx[,1])
image(1-heat.mat[ord,ord],col = heat.colors(100))

heat.mat[10,]
dat[c(10,29),]


pairs(dat2[sub.clust,2:14])

plot(apply(aa,1,function(x) length(unique(x))),type='l')
