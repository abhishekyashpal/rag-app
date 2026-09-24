class Perceptron:
    def __init__(self, learning_rate=0.1):
        self.w1 = 0.0
        self.w2 = 0.0
        self.b = 0.0
        self.learning_rate = learning_rate

    def predict(self, x1, x2):
        z = self.w1 * x1 + self.w2 * x2 + self.b

        if z >= 0:
            return 1
        else:
            return 0

    def train(self, x1, x2, y):
        prediction = self.predict(x1, x2)

        error = y - prediction

        self.w1 += self.learning_rate * error * x1
        self.w2 += self.learning_rate * error * x2
        self.b += self.learning_rate * error

        return prediction

def main():
    data = [
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 1)
    ] 

    model = Perceptron(learning_rate=0.1)

    for epoch in range(10):
        print(f"Epoch {epoch + 1}")
        for x1, x2, y in data:
            prediction = model.train(x1, x2, y)
            # print(
            #     "epoch:", epoch + 1,
            #     "w1:", model.w1,
            #     "w2:", model.w2,
            #     "b:", model.b
            # )ß
            print(
                x1, "AND", x2, "=", prediction, "(expected:", y, ")"
            )


    
if __name__ == "__main__":
    main()