import mnist_loader
import network
import time as time

training_data, validation_data, test_data =  mnist_loader.load_data_wrapper()

t0 = time.time()

net = network.Network([784, 30, 10])
net.SGD(training_data, 10, 3, 3.0, validation_data)
print("Training time: %.2f" % (time.time()-t0))
print("Testing set accuracy: %.2f%%" % (net.evaluate(test_data)/100.0))






