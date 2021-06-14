function activate = sigmoid(z)
	activate = 1.0 ./ (1.0 + (exp(-z)));
end