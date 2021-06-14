function [J grad] = costFunction(allTheta, X, y, n_hidden_units, n_output_units, lambda)
	% useful vars
	[m, n_input_units] = size(X);

	% get Theta1 & Theta2 from allTheta unrolled vector
	Theta1 = reshape(allTheta(1:(n_hidden_units * (n_input_units + 1))) ...
									, n_hidden_units, (n_input_units + 1));
	Theta2 = reshape(allTheta(((n_hidden_units * (n_input_units + 1)) + 1):end) ...
									, n_output_units, (n_hidden_units + 1));

	
	% compute hyp
	a1 = double([ones(m, 1), X]); % m x n_input_units+1
	z2 = a1 * Theta1'; % m x n_hidden_units
	a2 = sigmoid(z2);
	a2 = [ones(m, 1), a2]; % m x n_hidden_units+1
	z3 = a2 * Theta2'; % m x n_output_units
	a3 = hyp = sigmoid(z3); % m x n_output_units

	% compute hyp as a single output belonging to k units(0..9)
	%[max_act, hyp_x] = max(hyp, [], 2)


	% -- calculation of cost J --
	% recode y into m x n_output_units
	my_y = recode_y(y);

	error_func = (-my_y .* log(hyp)) - ((1 - my_y) .* log(1 - hyp));
	error_summation = sum(sum(error_func));

	regular_term_sum = sum(sum(Theta1(:, 2:end))) + sum(sum(Theta2(:, 2:end)));

	J = ((1/m) * error_summation) + ((lambda/(2*m)) * regular_term_sum);

	% -- backprop algo --
	delta3 = a3 - my_y; % m x n_output_units
	delta3 = delta3; % m x n_output_units

	delta2 = (delta3 * Theta2(:, 2:end)) .* sigmoid_deriv(z2); % m x n_hidden_units

	Theta1grad = delta2' * a1; % n_hid x n_inp+1
	Theta2grad = delta3' * a2; % n_out x n_hidden_units+1

	Theta1grad(:, 1) = (1/m) * Theta1grad(:, 1);
	Theta1grad(:, 2:end) = ((1/m) * Theta1grad(:, 2:end)) + ((lambda/m) * Theta1(:, 2:end) );

	Theta2grad(:, 1) = (1/m) * Theta2grad(:, 1);
	Theta2grad(:, 2:end) = ((1/m) * Theta2grad(:, 2:end)) + ((lambda/m) * Theta2(:, 2:end) );

	grad = [Theta1grad(:);Theta2grad(:)];

end