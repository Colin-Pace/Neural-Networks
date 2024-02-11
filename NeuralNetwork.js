class NeuralNetwork {
    constructor(inputs, hidden, outputs, showError = false, logFrequency, showSigmoidArgument = false) {
      this._inputs = inputs;
      this._hidden = hidden;
      this._outputs = outputs;
  
      this._bias0 = new Matrix(1, this._hidden);
      this._bias1 = new Matrix(1, this._outputs);
  
      this._weights0 = new Matrix(this._inputs, this._hidden);
      this._weights1 = new Matrix(this._hidden, this._outputs);
  
      this._bias0.randomWeights();
      this._bias1.randomWeights();
  
      this._weights0.randomWeights();
      this._weights1.randomWeights();
  
      this._logFrequency = logFrequency;
      this._iteration = 0;
  
      this._showError = showError;
      this._showSigmoidArgument = false;
    }
  
    get inputs() {
        return this._inputs;
    }
  
    set inputs(inputs) {
        this._inputs = inputs;
    }
  
    get hidden() {
        return this._hidden;
    }
  
    set hidden(hidden) {
        this._hidden = hidden;
    }
  
    get bias0() {
        return this._bias0;
    }
  
    set bias0(bias) {
        this._bias0 = bias;
    }
  
    get bias1() {
        return this._bias1;
    }
  
    set bias1(bias) {
        this._bias1 = bias;
    }
  
    get weights0() {
        return this._weights0;
    }
  
    set weights0(weights) {
        this._weights0 = weights;
    }
  
    get weights1() {
        return this._weights1;
    }
  
    set weights1(weights) {
        this._weights1 = weights;
    }
  
    feedForward(inputArray) {
        // convert input array to a matrix
        this.inputs = Matrix.convertFromArray(inputArray);
  
        // find the hidden values and apply the activation function
        this.hidden = Matrix.dot(this.inputs, this.weights0);
        this.hidden = Matrix.add(this.hidden, this.bias0); // apply bias
        this.hidden = Matrix.map(this.hidden, x => this.sigmoid(x), this._showSigmoidArgument);
  
        // find the output values and apply the activation function
        let outputs = Matrix.dot(this.hidden, this.weights1);
        outputs = Matrix.add(outputs, this.bias1); // apply bias
        outputs = Matrix.map(outputs, x => this.sigmoid(x), this._showSigmoidArgument);
  
        return outputs;
    }
  
    train(inputArray, targetArray) {
        // feed the input data through the network
        let outputs = this.feedForward(inputArray);
  
        // calculate the output errors (target - output)
        let targets = Matrix.convertFromArray(targetArray);
        let outputErrors = Matrix.subtract(targets, outputs);
  
        // error logs
        if (this._showError === true) {
          this._iteration++;
          if (this._iteration % this._logFrequency === 0) {
            console.log("Error: " + outputErrors.data[0][0]);
          }
        }
  
        // calculate the deltas (errors * derivitive of the output)
        let outputDerivs = Matrix.map(outputs, x => this.sigmoid(x, true), this._showSigmoidArgument);
        let outputDeltas = Matrix.multiply(outputErrors, outputDerivs);
  
        // calculate hidden layer errors (deltas "dot" transpose of weights1)
        let weights1T = Matrix.transpose(this.weights1);
        let hiddenErrors = Matrix.dot(outputDeltas, weights1T);
  
        // calculate the hidden deltas (errors * derivitive of hidden)
        let hiddenDerivs = Matrix.map(this.hidden, x => this.sigmoid(x, true), this._showSigmoidArgument);
        let hiddenDeltas = Matrix.multiply(hiddenErrors, hiddenDerivs);
  
        // update the weights (add transpose of layers "dot" deltas)
        let hiddenT = Matrix.transpose(this.hidden);
        this.weights1 = Matrix.add(this.weights1, Matrix.dot(hiddenT, outputDeltas));
        let inputsT = Matrix.transpose(this.inputs);
        this.weights0 = Matrix.add(this.weights0, Matrix.dot(inputsT, hiddenDeltas));
  
        // update bias
        this.bias1 = Matrix.add(this.bias1, outputDeltas);
        this.bias0 = Matrix.add(this.bias0, hiddenDeltas);
    }
  
    sigmoid(x, deriv = false, showSigmoidArgument) {
        if (showSigmoidArgument === true) {
          if (nn._iteration % nn._logFrequency === 0) {
            console.log("Sigmoid argument: " + x);
          }
        }
  
        if (deriv) {
            return x * (1 - x); // where x = sigmoid(x)
        }
        return 1 / (1 + Math.exp(-x));
    }
  }
  
  
  class Matrix {
    constructor(rows, cols, data = []) {
        this._rows = rows;
        this._cols = cols;
        this._data = data;
  
        // initialise with zeroes if no data provided
        if (data == null || data.length == 0) {
            this._data = [];
            for (let i = 0; i < this._rows; i++) {
                this._data[i] = [];
                for (let j = 0; j < this._cols; j++) {
                    this._data[i][j] = 0;
                }
            }
        } else {
            // check data integrity
            if (data.length != rows || data[0].length != cols) {
                throw new Error("Incorrect data dimensions!");
            }
        }
    }
  
    get rows() {
        return this._rows;
    }
  
    get cols() {
        return this._cols;
    }
  
    get data() {
        return this._data;
    }
  
    // add two matrices
    static add(m0, m1) {
        Matrix.checkDimensions(m0, m1);
        let m = new Matrix(m0.rows, m0.cols);
        for (let i = 0; i < m.rows; i++) {
            for (let j = 0; j < m.cols; j++) {
                m.data[i][j] = m0.data[i][j] + m1.data[i][j];
            }
        }
        return m;
    }
  
    // check matrices have the same dimensions
    static checkDimensions(m0, m1) {
        if (m0.rows != m1.rows || m0.cols != m1.cols) {
            throw new Error("Matrices are of different dimensions!");
        }
    }
  
    // convert array to a one-rowed matrix
    static convertFromArray(arr) {
        return new Matrix(1, arr.length, [arr]);
    }
  
    // dot product of two matrices
    static dot(m0, m1) {
        if (m0.cols != m1.rows) {
            throw new Error("Matrices are not \"dot\" compatible!");
        }
        let m = new Matrix(m0.rows, m1.cols);
        for (let i = 0; i < m.rows; i++) {
            for (let j = 0; j < m.cols; j++) {
                let sum = 0;
                for (let k = 0; k < m0.cols; k++) {
                    sum += m0.data[i][k] * m1.data[k][j];
                }
                m.data[i][j] = sum;
            }
        }
        return m;
    }
  
    // apply a function to each cell of the given matrix
    static map(m0, mFunction) {
        let m = new Matrix(m0.rows, m0.cols);
        for (let i = 0; i < m.rows; i++) {
            for (let j = 0; j < m.cols; j++) {
                m.data[i][j] = mFunction(m0.data[i][j]);
            }
        }
        return m;
    }
  
    // multiply two matrices (not the dot product)
    static multiply(m0, m1) {
        Matrix.checkDimensions(m0, m1);
        let m = new Matrix(m0.rows, m0.cols);
        for (let i = 0; i < m.rows; i++) {
            for (let j = 0; j < m.cols; j++) {
                m.data[i][j] = m0.data[i][j] * m1.data[i][j];
            }
        }
        return m;
    }
  
    // subtract two matrices
    static subtract(m0, m1) {
        Matrix.checkDimensions(m0, m1);
        let m = new Matrix(m0.rows, m0.cols);
        for (let i = 0; i < m.rows; i++) {
            for (let j = 0; j < m.cols; j++) {
                m.data[i][j] = m0.data[i][j] - m1.data[i][j];
            }
        }
        return m;
    }
  
    // find the transpose of the given matrix
    static transpose(m0) {
        let m = new Matrix(m0.cols, m0.rows);
        for (let i = 0; i < m0.rows; i++) {
            for (let j = 0; j < m0.cols; j++) {
                m.data[j][i] = m0.data[i][j];
            }
        }
        return m;
    }
  
    // apply random weights between -1 and 1
    randomWeights() {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] = Math.random() * 2 - 1;
            }
        }
    }
  }
  
  const inputs = 3;
  const hidden = 50;
  const outputs = 1;
  const iterations = 10000;
  const logFrequency = 1000;
  
  const nn = new NeuralNetwork(inputs, hidden, outputs);
  
  //   
  for (let i = 0; i < iterations; i++) {
    // Generate random input data and normalize
    let input0 = Math.random() * 2;
    let input1 = Math.random() * 2;
    let input2 = Math.random() * 2;

    const inclinationToGoUpward = 1.5;
    const inclinationAgainstGoingDownward = 0.25;
    let inputValues = [
        input0 * inclinationToGoUpward,
        input1,
        input2 * inclinationAgainstGoingDownward];
    let maxValue = Math.max(...inputValues);
    let indexOfMaxValue = inputValues.indexOf(maxValue);

    // Desired output
    let targetOutput = indexOfMaxValue / 2;
  
    // Feedforward and train in each iteration
    nn.train(inputValues, [targetOutput]);
    
    // Optionally, you can print the current error for monitoring
    // if (i % logFrequency === 0) {
    //   let currentError = nn.feedForward([input0, input1]);
    //   console.log(`Iteration ${i}, Error: ${currentError.data[0][0]}`);
    // }
  }
  
  // Test
  console.log("0, 1, 2 = " + nn.feedForward([0, 1, 2]).data);
  console.log("0, 2, 1 = " + nn.feedForward([0, 2, 1]).data);
  console.log("l, 0, 2 = " + nn.feedForward([1, 0, 2]).data);
  console.log("1, 2, 0 = " + nn.feedForward([1, 2, 0]).data);
  console.log("2, 0, 1 = " + nn.feedForward([2, 0, 1]).data);
  console.log("2, 1, 0 = " + nn.feedForward([2, 1, 0]).data);
