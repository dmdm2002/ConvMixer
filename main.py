from Utils.Options import Param
from Runner.Train import Trainer


class driver(Param):
    def __init__(self):
        super(driver, self).__init__()

    def run_train(self):
        tr = Trainer()
        tr.run()

    def __call__(self, *args, **kwargs):
        if self.run_type == 0:
            return self.run_train()


if __name__ == "__main__":
    driver()()