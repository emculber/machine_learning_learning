function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

% Input 400 nodes 20x20 pixels
% Hidden 25 nodes
% nn_params 10285x1 1-10025
% Theta1 1    -10025 25x401 matrix
% Theta2 10026-10285 10x26  matrix
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
         
J = 0;

tridelta1 = 0;
tridelta2 = 0;

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Feedforward:
y_eye = eye(num_labels);
y = y_eye(y, :); % convert y 10x1 matrix 1 the value else 0

a1 = [ones(m, 1) X]; % Input add bias
z2 = a1 * Theta1';   % hidden layer
a2 = [ones(m, 1) sigmoid(z2)]; % Hidden layer activation add bias
z3 = a2 * Theta2'; % Output layer
a3 = sigmoid(z3); % Output layer activation

error_befor_sum = (-y).*log(a3)-(1-y).*log(1-a3); % inner cost function
%J = (1/m).*sum(sum(error_befor_sum)); % cost function without regression

Theta1_without_bias = Theta1(:, 2:end); % remove bias
Theta2_without_bias = Theta2(:, 2:end); % remove bias
% Cost function with regression
J = ((1/m).*sum(sum(error_befor_sum)))+(lambda/(2*m)).*(sum(sum(Theta1_without_bias.^2))+sum(sum(Theta2_without_bias.^2)));

% BackPropagation
delta3 = (a3-y);
z2=[ones(m,1) z2];
delta2 = delta3*Theta2.*(sigmoid(z2).*(1-sigmoid(z2)));
delta2 = delta2(:, 2:end);
tridelta1 = tridelta1 + delta2'*a1;
tridelta2 = tridelta2 + delta3'*a2;

Theta1_grad = (1/m).*tridelta1;
Theta2_grad = (1/m).*tridelta2;

Theta1_grad(:, 2:input_layer_size+1) = Theta1_grad(:, 2:input_layer_size+1) + lambda / m * Theta1(:, 2:input_layer_size+1);
Theta2_grad(:, 2:hidden_layer_size+1) = Theta2_grad(:, 2:hidden_layer_size+1) + lambda / m * Theta2(:, 2:hidden_layer_size+1);

%Theta1_grad = (1/m).*tridelta1 + (lambda/m).*Theta1;
%Theta2_grad = (1/m).*tridelta2 + (lambda/m).*Theta2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
