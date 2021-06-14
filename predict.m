function pred = predict(X, allTheta, n_hidden_units, n_output_units)
	[m, n_input_units] = size(X);

	Theta1 = reshape(allTheta(1:(n_hidden_units * (n_input_units + 1))) ...
									, n_hidden_units, (n_input_units + 1));
	Theta2 = reshape(allTheta(((n_hidden_units * (n_input_units + 1)) + 1):end) ...
									, n_output_units, (n_hidden_units + 1));

	a1 = double([ones(m, 1), X]); % m x n_input_units+1
	z2 = a1 * Theta1'; % m x n_hidden_units
	a2 = sigmoid(z2);
	a2 = [ones(m, 1), a2]; % m x n_hidden_units+1
	z3 = a2 * Theta2'; % m x n_output_units
	a3 = hyp = sigmoid(z3); % m x n_output_units

	[maxVal, hyp] = max(hyp, [], 2);

	pred = hyp';

end