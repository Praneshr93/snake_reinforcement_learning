import numpy as np
import tensorflow as tf
import random

# Constants
SEED = 0
TAU = 0.01  # soft update parameter
ALPHA = 1e-3  # learning rate
snake_optimizer = tf.keras.optimizers.Adam(learning_rate=ALPHA)
random.seed(SEED)


# Functions
def get_experiences(memory_buffer, minibatch_size, board_width):
    """
    This function samples random experiences from the memory buffer
    and returns them as expereince tuples
    """
    experiences = random.sample(memory_buffer, k=minibatch_size)
    states = tf.convert_to_tensor(
        np.array([np.reshape(e.state, (board_width, board_width, 3)) for e in experiences if e is not None]),
        dtype=tf.float32
    )
    actions = tf.convert_to_tensor(
        np.array([e.action for e in experiences if e is not None]),
        dtype=tf.float32
    )
    rewards = tf.convert_to_tensor(
        np.array([e.reward for e in experiences if e is not None]),
        dtype=tf.float32
    )
    next_states = tf.convert_to_tensor(
        np.array([np.reshape(e.next_state, (board_width, board_width, 3)) for e in experiences if e is not None]),
        dtype=tf.float32
    )
    done_vals = tf.convert_to_tensor(
        np.array([e.done for e in experiences if e is not None]),
        dtype=tf.float32
    )
    return (states, actions, rewards, next_states, done_vals)


def compute_loss(experiences, gamma, q_network, target_q_network):
    """
    This function calculates the target values for the state-action pair
    and returns the error with respect the model predictions
    """
    # Unpack the mini-batch of experience tuples
    states, actions, rewards, next_states, done_vals = experiences

    # Compute max Q^(s,a) - the ideal action selected using target-Q-Network
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)

    # Set the reference Q-values for the states using target-Q-Network
    # y = R if episode terminates, otherwise set y = R + γ max Q^(s,a)
    y_targets = rewards + ((1 - done_vals) * gamma * max_qsa)

    # Get the q_values
    q_values = tf.reduce_max(q_network(states), axis=-1)

    # Compute the loss
    loss = tf.keras.losses.MSE(y_targets, q_values)

    return loss


def update_target_network(q_network, target_q_network):
    """
    This function performs soft updates on target-Q-Network weights
    """
    for target_weights, q_net_weights in zip(
        target_q_network.weights, q_network.weights
    ):
        target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)


@tf.function
def agent_learn(experiences, gamma, q_network, target_q_network):
    """
    This function performs one step of gradient descent from sampled experience
    """
    # Calculate the loss
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)

    # Get the gradients of the loss with respect to the weights
    gradients = tape.gradient(loss, q_network.trainable_variables)

    # Update the weights of the q_network
    snake_optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

    # update the weights of target q_network
    update_target_network(q_network, target_q_network)
