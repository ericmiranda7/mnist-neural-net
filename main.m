% dataset
d = load('mnist.mat');

% useful vars
options = optimset('MaxIter', 100, 'GradObj', 'on');
lambda = 1;

n_hidden_units = 100;
n_output_units = 10;
[m, n_input_units] = size(d.trainX);

n_rows = (n_hidden_units * (n_input_units + 1)) + ( n_output_units * (n_hidden_units + 1) );

% guess random theta
initialTheta = zeros(n_rows, 1);
initialTheta = rand_init(initialTheta);

costFunc = @(p) costFunction(p, d.trainX, d.trainY, n_hidden_units, n_output_units, lambda);

[optimTheta, cost] = fmincg(costFunc, initialTheta, options);

