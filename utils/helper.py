class AverageTracker():

    def __init__(self):
        self.total = 0.0
        self.counter = 0
        self.avg = 0.0

    def update(self, val, count=1):

        self.total += val
        self.counter += count
        self.avg = self.total / self.counter


class Progress:

    def __init__(self, header) -> None:
        pass

    def update(self, prefix):

        print(f"{prefix}", end="\r")

    def close(self):
        print("")


class ConfigStruct:

    def __init__(self, **args) -> None:
        self.__dict__.update(**args)