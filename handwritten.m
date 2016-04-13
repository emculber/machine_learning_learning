clear ; close all; clc

input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10 (mapped "0" to label 10)

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')
load('handwritten_data.mat');
m = size(X, 1);           % Number of training sets

% Randomly select 100 data points to display
randomize_set = randperm(size(X, 1));
random_set = randomize_set(1:100);
displayData(X(random_set, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nInitializing Neural Network Parameters ...\n')

rand("seed", 0)
epsilon_init = 0.12;
initial_Theta1 = rand(hidden_layer_size, 1+input_layer_size) * 2 * epsilon_init - epsilon_init;
initial_Theta2 = rand(num_labels, 1+hidden_layer_size) * 2 * epsilon_init - epsilon_init;

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 500);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
