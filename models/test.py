from AlexNet import AlexNet
from LeNet import LeNet

def test():
    model = AlexNet(0.001, 10)
    model = LeNet(0.001, 10)
    print('All tests passed')

if __name__ == '__main__':
    test()
