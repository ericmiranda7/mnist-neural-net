load optimTheta;
d = load('mnist.mat');

preds = predict(d.testX, optimTheta, n_hidden_units, n_output_units);

my_y = d.testY;
for i = 1:10000
	if (my_y(i) == 0)
		my_y(i) = 10;
	end
end

res = my_y == preds;
acc = sum(res(:) == 1);
acc = (acc / 10000) * 100;