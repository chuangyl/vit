class AverageMeter():
    def __init__(self) -> None:
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.reset()
    
    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
    
    def update(self,val,n=1):
        self.sum += val*n
        self.cnt += n
        self.avg = self.sum/self.cnt