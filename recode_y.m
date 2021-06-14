function res = recode_y(y)
	% recode y into m x n_output_units
	y = y'; % m x 1
	m = size(y, 1);
	recoded_y = zeros(m, 10); % m x n_output_units
	for i = 1:m
		curr_y = y(i);
		if(curr_y == 0);
			y(i) = 10;
		end

		recoded_y(i, y(i)) = 1;
		
	end

	res = recoded_y;
end