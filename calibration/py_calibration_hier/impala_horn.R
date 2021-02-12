dat<-read.csv('~/git/impala/calibration/py_calibration_hier/results/Ti64/res_ti64_hier3_impalaHorn_postSamplesPTW.csv',skip=1)

hist(dat$stress,xlab='posterior stress', main='',breaks=50)
legend('topright',legend=c('strain = 0.6', 'edot = 40250/s', 'temp = 694K'),bty='n')
abline(v=quantile(dat$stress,probs=c(.025,.975)),lwd=3,lty=2)
low<-which(dat$rank==2000*.05)
up<-which(dat$rank==2000*.95)
low<-which(dat$rank<10)
up<-which(dat$rank>1990)

quantile(dat$stress,probs=c(.1,.9))

rbPal <- colorRampPalette(c('dodgerblue','gold'))

#This adds a column of color values
# based on the y values
col <- rbPal(100)[as.numeric(cut(dat$stress,breaks = 100))]
cex<-rep(.1,2000)

col[low]<-'purple'#'darkblue'
col[up]<-'red'#'maroon'
cex[low]<-1
cex[up]<-1

ord<-c((1:2000)[-c(low,up)],low,up)

pdf('~/Desktop/extreme10-2.pdf',height=10,width=10)
pairs(dat[ord,1:11],col=col[ord],cex=cex[ord],lower.panel = NULL)
dev.off()
