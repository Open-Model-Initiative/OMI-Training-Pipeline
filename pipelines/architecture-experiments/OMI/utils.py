class RunningMean:
    def __init__(self):
        self.count=0
        self.value=0.0

    def update(self,value):
        self.count+=1
        self.value+=value

    def get(self):
        return self.value/self.count