require(ggplot2)
require(ggrepel)
require(Cairo)
require(tikzDevice)

entropies <- read.csv('group_predictability.csv',sep=';',header=F)

#entropies$V2 <- rank(entropies$V2)
#entropies$V3 <- rank(entropies$V3)


tikz('unpredictability_graph')
#ggplot() + geom_text_repel(data=entropies,aes(x=V3,y=V2,label=V1),cex=3) + labs(x='unpredictability of cluster given form',y='unpredictability of form given cluster')
ggplot() +
  geom_point(data=entropies,aes(x=V3,y=V2),cex=.5,alpha=.25) +
  geom_text_repel(data=entropies,aes(x=V3,y=V2,label=V1),cex=3,segment.alpha=.25) + labs(x='unpredictability of cluster label given form',y='unpredictability of form given cluster label')

dev.off()