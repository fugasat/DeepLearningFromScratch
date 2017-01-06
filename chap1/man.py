class Man:
    def __init__(self, name):
        self.name = name
        print("Initialized to " + name)

    def hello(self):
        print("Hello " + self.name + "!")

    def goodbye(self):
        print("Goodbye " + self.name + "!")


class WorkerMan(Man):
    def __init__(self, name, workerId):
        super().__init__(name)
        self.workerId = workerId
        print("Worker initialized to " + name + ":" + str(workerId))

    def work(self):
        print("Working " + self.name + "!")

    # override
    def goodbye(self):
        print("Goodbye worker " + self.name + "!")


man = Man("David")
man.hello()
man.goodbye()

worker = WorkerMan("Tom", 12345)
worker.hello()
worker.work()
worker.goodbye()
