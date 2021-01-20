import numpy

class Test():
    def __init__(self):
        self.count = 0
        self.sample_csi = []
        self.new_csi =numpy.zeros((50,3,3))

    def test(self, csi, **kwargs):

        print('start test')
        self.count = self.count + 1
        self.sample_csi.append(csi)
        print(self.sample_csi)
        if self.count == 5:
            print('1')
            new_csi = numpy.array(self.sample_csi)
            print(new_csi)
            self.new_csi = numpy.concatenate(new_csi, axis = 2).transpose(2, 0, 1)

            self.sample_csi =[]
            self.count = 0

        return self.new_csi
if __name__ == '__main__':
    c = Test()
    a = numpy.ones((3,3,10))
    b = c.test(a)
    b = c.test(a)
    b = c.test(a)
    b = c.test(a)
    b = c.test(a)
    print(b)