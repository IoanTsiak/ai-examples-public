"""Learn the XOR truth table with a tiny neural network in PyTorch.

HOW TO RUN
1. Save this file as XOR_Beginner_Lesson.py.
2. In a terminal, install PyTorch if needed:
       python -m pip install torch
       %pip -q install torch (collab)
3. Run the lesson:
       python XOR_Beginner_Lesson.py

A terminal is an application where you type commands to run programs.
This lesson runs locally on your computer's CPU. It needs no API key.

THE BIG IDEA
We give a model four examples and their correct answers. The model starts
with random settings. Training repeatedly adjusts those settings to make
its predictions closer to the correct answers.

This is supervised learning: each training example has a known answer.
We train a new, tiny model from scratch. Fine-tuning would mean starting
with a model that has already been trained and adjusting it with more data.

PYTHON BASICS
Lines beginning with # are comments. Python does not execute them.
This opening block between triple quotes is a description, or docstring.
An assignment such as training_steps = 1000 gives a name to a value.
Indentation groups instructions inside a loop or a with block.
"""

# -----------------------------------------------------------------------------
# 0. Load the tools and choose repeatable starting settings
# -----------------------------------------------------------------------------

# A library is a collection of reusable code.
# PyTorch is a library for working with numbers and building neural networks.
# import makes that library available under the name torch.
import torch

# nn is PyTorch's neural-network toolkit. It contains layers and loss functions.
from torch import nn

# A neural network begins with randomly chosen weights and biases.
# A seed chooses the starting point of a pseudo-random number sequence.
# The number 7 is arbitrary. It is not a learning rate or a quality setting.
# Using the SAME seed and the same operations in the same environment helps
# repeat the same starting values. Different seeds give different sequences.
# manual_seed seeds PyTorch's generators on the CPU and other devices.
# Exact reproducibility across versions and platforms is not guaranteed.
torch.manual_seed(7)

# A CPU thread is a stream of work handled by the processor.
# One thread is sufficient for this tiny demonstration. This setting alone
# does not guarantee deterministic results. It does not change the XOR rule.
torch.set_num_threads(1)


# -----------------------------------------------------------------------------
# 1. Create the examples the network should learn
# -----------------------------------------------------------------------------

# XOR means "exactly one of the two inputs is 1."
# Its truth table lists every possible pair and the correct result:
#     0 XOR 0 = 0
#     0 XOR 1 = 1
#     1 XOR 0 = 1
#     1 XOR 1 = 0
# These are the four examples we want our model to learn.

# A tensor is a container of numbers, similar to a list or a table.
# Here, each row is an example. Each column is an input feature.
# A feature is a piece of information the model receives to make a prediction.
# Our tensor has shape (4, 2): four examples, each containing two numbers.
# We use decimals because the network performs calculations with real numbers.
inputs = torch.tensor(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ],
    dtype=torch.float32,  # Store numbers using 32-bit floating-point precision.
)

# A label (also called a target) is the correct answer for an example.
# The row order must match inputs. For example, the second input [0, 1]
# has the second label [1]. The labels have shape (4, 1).
# The inner brackets keep one column, matching the model's output shape.
# This loss function expects floating-point labels, even for classes 0 and 1.
correct_answers = torch.tensor(
    [
        [0.0],
        [1.0],
        [1.0],
        [0.0],
    ],
    dtype=torch.float32,
)

# print displays text. zip pairs rows from two containers in their given order.
# The for loop repeats its indented instructions for each pair of rows.
# tolist() converts a tensor row to an ordinary Python list.
# item() extracts the number from a tensor containing just one value.
# int() displays that value as a whole number, such as 1 instead of 1.0.
# An f-string starts with f and inserts the values inside {...} into the text.
print("Training examples:")
for input_pair, answer in zip(inputs, correct_answers):
    print(f"{input_pair.tolist()} -> {int(answer.item())}")


# -----------------------------------------------------------------------------
# 2. Build a small neural network
# -----------------------------------------------------------------------------

# A neural network is a mathematical function with adjustable parameters.
# Parameters are the numbers training changes: weights and biases.
# A weight controls how strongly an input affects a calculation.
# A bias is an extra adjustable number added to that calculation.

# A linear-layer neuron calculates a weighted sum plus a bias:
#     value = (weight_1 * input_1) + (weight_2 * input_2) + bias
# For illustration, if weights are 0.4 and -0.2, bias is 0.1, and inputs
# are [1, 0], the result is (0.4 * 1) + (-0.2 * 0) + 0.1 = 0.5.
# These example parameters are illustrative. PyTorch chooses our initial ones.

# A single linear layer cannot correctly separate XOR's two classes.
# Imagine plotting the four inputs as points on a square. The two class-1
# points lie on opposite corners. No single straight line separates them
# from both class-0 points.

# Our network has three stages:
#     two input features -> four hidden values -> one output score
# A hidden layer is an intermediate calculation between input and output.
# Four is a design choice, not the number of labels or training examples.

# Sequential sends the output of each stage into the next stage, in order.
model = nn.Sequential(
    # Each of four neurons reads both inputs and has its own weights and bias.
    # A batch of shape (4, 2) becomes a batch of shape (4, 4).
    nn.Linear(in_features=2, out_features=4),

    # An activation function transforms the hidden values.
    # Tanh maps each value into the range between -1 and 1.
    # For example, tanh(0) = 0 and tanh(0.5) is approximately 0.462.
    # This nonlinear transformation lets the network represent XOR.
    # Without it, two linear layers still act like one linear transformation.
    nn.Tanh(),

    # Combine the four hidden values into one raw score for each example.
    # The output batch has shape (4, 1), matching correct_answers.
    # This raw score is called a logit. It is not yet a probability.
    nn.Linear(in_features=4, out_features=1),
)

# A loss is a number measuring disagreement between predictions and labels.
# Smaller loss means a better fit according to the chosen loss function.
# Binary classification means choosing between two classes, here 0 and 1.
# BCEWithLogitsLoss combines sigmoid and binary cross-entropy in a way that
# helps avoid numerical problems. Pass RAW SCORES to it, not probabilities.
# It averages the four examples' losses by default.
# For one example with label 1, predicted probability 0.9 gives loss about
# 0.105, while probability 0.1 gives loss about 2.303: a confident wrong
# prediction is penalized more. Loss is not a percentage of wrong answers.
loss_function = nn.BCEWithLogitsLoss()

# An optimizer is the rule used to adjust the parameters during training.
# Adam uses gradients and information from previous steps to choose updates.
# model.parameters() supplies all the weights and biases to the optimizer.
# lr means learning rate: it controls the scale of those updates.
# A large learning rate can make learning unstable. A small one can be slow.
# 0.1 is a choice for this toy example, not a universal recommended value.
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


# -----------------------------------------------------------------------------
# 3. Train on the four examples
# -----------------------------------------------------------------------------

# One training step here uses all four examples and makes one parameter update.
# We repeat this process 1000 times. Since every step uses the entire dataset,
# each step also represents one pass through the data, often called an epoch.
training_steps = 1000

# [] creates an empty Python list. We will store the loss values in it.
# Keeping this history is optional and could help us plot learning later.
loss_history = []

# Set training mode. This matters for certain layers, such as dropout.
# Our Linear and Tanh layers behave the same in training and evaluation modes.
model.train()

# range(1, 1001) produces 1 through 1000. The upper limit is excluded.
for step in range(1, training_steps + 1):
    # A gradient describes how loss changes when a parameter changes slightly.
    # PyTorch accumulates gradients unless we clear them. Clear the old ones
    # so this update uses gradients from the current calculation.
    optimizer.zero_grad()

    # FORWARD PASS: calculate predictions using the current parameters.
    # All four input rows go through the network together as one batch.
    # The model receives inputs only. It does not receive the correct labels.
    predicted_scores = model(inputs)

    # MEASURE ERROR: compare those raw scores with the known labels.
    # PyTorch records the calculations needed to work out gradients later.
    loss = loss_function(predicted_scores, correct_answers)

    # Extract a plain Python number and append it to the loss history.
    # This stored value describes the model BEFORE this step's update.
    loss_history.append(loss.item())

    # BACKWARD PASS: calculate gradients for the weights and biases.
    # This is backpropagation. PyTorch follows the calculation backward and
    # applies the chain rule from calculus. You do not need to derive it here.
    # backward() calculates gradients. It does not itself update parameters.
    loss.backward()

    # UPDATE: Adam uses those gradients to change weights and biases.
    # The aim is to reduce loss. A single update is not guaranteed to do so.
    # Across many successful steps, predictions should approach the labels.
    optimizer.step()

    # % gives the remainder after division. step % 200 == 0 means the step
    # is a multiple of 200. == compares values, while = assigns a value.
    # Print at step 1 and then at 200, 400, 600, 800, and 1000.
    # :4d prints an integer in a field four characters wide.
    # :.4f prints a decimal number with four digits after the decimal point.
    if step == 1 or step % 200 == 0:
        print(f"Training step {step:4d} | loss: {loss.item():.4f}")


# -----------------------------------------------------------------------------
# 4. Check the trained model and understand its probabilities
# -----------------------------------------------------------------------------

# Evaluation mode marks that we are using the model for predictions.
# It does not erase training, turn off gradients, or freeze weights by itself.
model.eval()

# no_grad temporarily stops recording calculations for gradient computation.
# We only want predictions here, so this saves unnecessary work and memory.
# The with block applies this setting to the indented instructions inside it.
with torch.no_grad():
    # Make fresh predictions AFTER the final training update.
    final_scores = model(inputs)

    # Sigmoid converts a logit z into a number between 0 and 1:
    #     probability = 1 / (1 + exp(-z))
    # exp means the exponential function. You do not need to calculate it
    # manually: torch.sigmoid applies this formula to every output score.
    # Some illustrative values, not promised outputs from this trained model:
    #     logit -2 -> probability about 0.119 -> 11.9% for class 1
    #     logit  0 -> probability       0.500 -> 50.0% for class 1
    #     logit  2 -> probability about 0.881 -> 88.1% for class 1
    # Training changes the weights, which changes logits and probabilities.
    probabilities = torch.sigmoid(final_scores)

    # We interpret these values as the model's estimated probability of class 1.
    # For example, 0.881 gives 88.1% for class 1 and 11.9% for class 0.
    # A large probability is not a guarantee that the model is correct.

    # A threshold converts the continuous estimate into a discrete answer.
    # >= 0.5 gives True for class 1 and False for class 0.
    # int() on this tensor converts True to 1 and False to 0.
    # The threshold is our decision rule. It does not change the probabilities.
    predicted_answers = (probabilities >= 0.5).int()

# \n inserts a new line. zip groups matching rows from all four tensors.
print("\nResults after training:")
for input_pair, expected, probability, prediction in zip(
    inputs, correct_answers, probabilities, predicted_answers
):
    # Show the input, correct label, predicted label, and estimated probability.
    # Multiplying by 100 converts a probability to a percentage.
    print(
        f"{input_pair.tolist()} -> expected {int(expected.item())}, "
        f"predicted {int(prediction.item())}, "
        f"probability of 1: {probability.item():.3f} "
        f"({probability.item() * 100:.1f}%)"
    )


# -----------------------------------------------------------------------------
# 5. Student experiments and what this demonstration proves
# -----------------------------------------------------------------------------

# Change one setting at a time, rerun the ENTIRE script, and compare results.
# Rerunning from the start builds a fresh model instead of continuing training.
# - Change 4 hidden neurons to 2 or 8. Update BOTH Linear layers to match.
# - Change training_steps to 100 or 2000. How does the final loss change?
# - Change lr to 0.01 or 0.5. Does learning become slower or less stable?
# - Change the seed from 7 to 12. Do the starting loss and final results change?
# - Try AND labels by replacing correct_answers with:
#       torch.tensor([[0.0], [0.0], [0.0], [1.0]])
# - Try OR labels by replacing correct_answers with:
#       torch.tensor([[0.0], [1.0], [1.0], [1.0]])
# - Remove nn.Tanh() and rerun with XOR. Why is a linear model insufficient?

# We checked the SAME four examples used for training. This shows how well
# the model fitted this complete binary truth table, not performance on
# independent data. Predictions for inputs such as [0.3, 0.7] are outside
# the binary XOR examples and have no specified correct answer in this lesson.
# Larger projects usually evaluate on separate examples not used for training.
# This toy network demonstrates learning. It cannot answer arbitrary questions.

# Official background reading:
# https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
# https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
# https://docs.pytorch.org/docs/stable/notes/randomness.html
