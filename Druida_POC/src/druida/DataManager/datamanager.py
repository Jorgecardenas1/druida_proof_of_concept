class dataSets:
    def __init__(self, name, size, batch_size):
        self.name = name
        self.size=size
        self.batch_size =  batch_size

    def create(self):
        print(str(self.name))

    

