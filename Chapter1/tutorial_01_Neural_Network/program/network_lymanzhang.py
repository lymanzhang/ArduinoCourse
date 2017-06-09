# TODO: implement a Neutal Network
import Network(object):
    def _init_(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.sample((y,1)) for y in sizes[1:]]
        self.weights = [np.random.sample((y,x)) for x, y in zip(sizes[:1], size[1:])]

    def SGD(self, tr_data, epochs, batch_size, alpha, val_data):
        "Stochastic gradient descent"
        n = len(tr_data)
        for j in range(epochs):
            np.random.shuffle(tr_data)
            batches = [tr_data[k:k+batch_size] for k in range(0,n,batch_size)]
            for batch in batches:
                self.update_parameters(batch, batch_size, alpha)
            print('Epoch %d %d / %d'%(j,self.evaluate(val_data)), len(val_data))

        def update_parameters(self, batch,batch_size):
            X,y = np.array(batch).T
            delta = np.array(list(map(self.backprob,X,y)))
            delta_b,delta_w = np.sum(delta,axis=0)/bach_siez*alpha
            self.weights = np.subtract(self.weights, delta_w)
            self.biases = np.subtract(self.biases, delta_b)

        def backprob(self,X,y):
            "Backporpagation"
            mini_dalta_b = [np.zeros(b.shape) for b in self.biases]
            mini_dalta_w = [np.zeros(w.shape) for w in self.weights]

        #feedforward
            activation = X
            activations = [X]
            zx = []
            for b,w in zip(self.biases, self.weights):
                z = np.dot(w,activation) + b
                zx.append(z)
                activation = sigmoid(z)
                activation.append(activation)

        #backward
            delta = self.cost_derivative(activation[-1],y)*sigmoid_prime(zs[-1])
            mini_delta_b[-1] = delta
            mini.delta_w[-1] = np.dot(delta,activations[-2]).T

            for l in range(2,self.num_layers):
                z= zs[-1]
                sp = sigmoid_prime[z]
                delta = np.dot(self.weights[-l+1].T,delta)*sp
                mini_delta_b[-1] = delta
                mini.delta_w[-1] = np.dot(delta,activations[-l-1]).T
            return (mini_delta_b, mini.delta_w)

        def evaturate(self, test_data):
            test_tesults = [np.argmax(self.feedforward(x)),y for x,y in test_data]
            return sum(int(x==y) for (x,y) in test_tesults)

        def cost_derivative(self,output_activations,y):
            return (output_activations-y)

        def sigmoid(x):
            