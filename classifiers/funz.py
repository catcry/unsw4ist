# Some Usefull Functions
"""
______________________________________________________________________________
Some Useful  functions :
    > Calculating competency metrics from confusion matrix    
    > number (n) of in (x)
______________________________________________________________________________

@author: catcry
"""

#==============================================================================
#------------------->>>    Getting Metrics from Conf Matrix    <<<-------------
#==============================================================================

class Metrix():
    def __init__ (self,cm):
        self.cm = cm
        self.ac = ((self.cm[0,0]+self.cm[1,1]) / \
                   (self.cm[0,0]+self.cm[0,1]+self.cm[1,0]+self.cm[1,1]))*100
        self.dr = (self.cm[1,1] / (self.cm[1,1] + self.cm[1,0]))*100
        self.pr = (self.cm[1,1] / (self.cm[1,1] + self.cm[0,1]))*100
        self.fpr = (self.cm[0,1] / (self.cm[0,0] + self.cm[0,1]))*100
        self.fm = ((2*self.dr*self.pr) / (self.dr+self.pr))*100
        self.fr = (self.cm[1,0]/(self.cm[0,0]+self.cm[1,0]))*100
    

______________________________________________________________________________
def num_of(n,x):
    p=0
    num_of_n=0
    while p<len(x):
        if x[p]==n:
                num_of_n+=1
        p+=1
    return num_of_n

def rnd(x):
    i=0
    while i<len(x):
        if x[i] < 0.5:
            x[i]=0
        else:
            x[i]=1
        i+=1
    return x


#%%
