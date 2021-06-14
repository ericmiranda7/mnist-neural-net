function [rand_theta] = rand_init(Theta)
	EPS_INIT = e^-4;
	
	rand_theta = (rand(size(Theta)) * (2*EPS_INIT)) - EPS_INIT;
end