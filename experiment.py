import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import string as sn
from time import time
from datetime import datetime


# The location on the disk of project
PROJECT_BASEDIR = ("G:/My Drive/Academic/Research/" + 
    "Program Synthesis with Deep Learning/")


# The location on the disk of plots and images
PLOT_BASEDIR = (PROJECT_BASEDIR + "Results/")


# The location on the disk of checkpoints
CHECKPOINT_BASEDIR = (PROJECT_BASEDIR + "Backups/")


# The location on the disk of project
DATASET_BASEDIR = (PROJECT_BASEDIR + "Datasets/")


# Filenames associated with program dataset
DATASET_FILENAMES = [
    (DATASET_BASEDIR + "epf_5_dataset.csv")
]


# Locate dataset files on hard disk
for FILE in DATASET_FILENAMES:
    if not tf.gfile.Exists(FILE):
        raise ValueError('Failed to find file: ' + FILE)


# Dataset configuration constants
DATASET_IO_EXAMPLES = 10
DATASET_COLUMNS = (DATASET_IO_EXAMPLES * 2) + 2
DATASET_DEFAULT = "0"
DATASET_MAXIMUM = 64
DATASET_VOCABULARY = sn.printable
VOCAB_SIZE = len(DATASET_VOCABULARY)
TOTAL_MUTATIONS = 10


# Batch and logging configuration constants
BATCH_SIZE = 32
TOTAL_EXAMPLES = 9330
EPOCH_SIZE = TOTAL_EXAMPLES // BATCH_SIZE
TOTAL_LOGS = 100


# Prefix nomenclature
PREFIX_TOTAL = "total"
PREFIX_GENERATOR = "generator"
PREFIX_SYNTAX = "syntax"
PREFIX_BEHAVIOR = "behavior"


# Extension nomenclature
EXTENSION_NUMBER = (lambda number: "_" + str(number))
EXTENSION_LOSS = "_loss"
EXTENSION_WEIGHTS = "_weights"
EXTENSION_BIASES = "_biases"


# Collection nomenclature
COLLECTION_LOSSES = "_losses"
COLLECTION_PARAMETERS = "_parameters"
COLLECTION_ACTIVATIONS = "_activations"


# LSTM structural hyperparameters
ENSEMBLE_SIZE = 1
LSTM_SIZE = (len(DATASET_VOCABULARY) * 2 * ENSEMBLE_SIZE)
DROPOUT_PROBABILITY = (1 / ENSEMBLE_SIZE)
USE_DROPOUT = True


# Behavior function hyperparameters
BEHAVIOR_WIDTH = 4
BEHAVIOR_DEPTH= 1
BEHAVIOR_TOPOLOGY = ([[1, BEHAVIOR_WIDTH]] + 
    [[BEHAVIOR_WIDTH, BEHAVIOR_WIDTH] for i in range(BEHAVIOR_DEPTH)] + 
    [[BEHAVIOR_WIDTH, 1]])


# Training hyperparameters
INITIAL_LEARNING_RATE = 0.001
DECAY_STEPS = EPOCH_SIZE
DECAY_FACTOR = 0.5


# Precision recall threshold parameters
THRESHOLD_UPPER = 1.0
THRESHOLD_LOWER = 0.0
THRESHOLD_RANGE = 100
THRESHOLD_DELTA = (THRESHOLD_UPPER - THRESHOLD_LOWER) / THRESHOLD_RANGE
DENSITY_FUNCTION = (lambda x: (6 / (THRESHOLD_UPPER - THRESHOLD_LOWER)**3) * (x - THRESHOLD_LOWER) * (THRESHOLD_UPPER - x))
SAMPLE_WEIGHT = (lambda x: DENSITY_FUNCTION(x) * THRESHOLD_DELTA)


# Convert elements of python source code to one-hot token vectors
def tokenize_program(program, vocabulary=DATASET_VOCABULARY):

    # List allowed characters
    mapping_characters = tf.string_split([vocabulary], delimiter="")


    # List characters in each word
    input_characters = tf.string_split([program], delimiter="")


    # Create integer lookup table
    lookup_table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_characters.values, default_value=0)


    # Query lookup table
    one_hot_tensor = tf.one_hot(lookup_table.lookup(input_characters.values), len(vocabulary), dtype=tf.float32)


    # Calculate actual sequence length
    actual_length = tf.size(one_hot_tensor) // len(vocabulary)


    # Pad input to match DATASET_MAXIMUM
    expanded_tensor = tf.pad(one_hot_tensor, [[0, (DATASET_MAXIMUM - actual_length)], [0, 0]])

    return tf.reshape(expanded_tensor, [DATASET_MAXIMUM, len(vocabulary)]), actual_length


def detokenize_program(program, vocabulary=DATASET_VOCABULARY):

    # List allowed characters
    mapping_characters = tf.string_split([vocabulary], delimiter="")


    # Select one hot index
    indices = tf.argmax(program, axis=2)


    # Create integer lookup table
    lookup_table = tf.contrib.lookup.index_to_string_table_from_tensor(mapping_characters.values, default_value=DATASET_DEFAULT)


    # Lookup corrosponding string
    program = lookup_table.lookup(indices)

    return program


# Read single row words
def decode_record(filename_queue, num_columns=DATASET_COLUMNS, default_value=DATASET_DEFAULT):

    # Attach text file reader
    DATASET_READER = tf.TextLineReader(skip_header_lines=1)


    # Read single line from dataset
    key, value_text = DATASET_READER.read(filename_queue)


    # Decode line to columns of strings
    name_column, *example_columns, function_column = tf.decode_csv(value_text, [[default_value] for i in range(num_columns)])


    # Convert IO examples from string to float32
    example_columns = tf.string_to_number(tf.stack(example_columns), out_type=tf.float32)


    # Convert python code to tokenized one-hot vectors
    program_tensor, actual_length = tokenize_program(function_column)

    return name_column, example_columns, program_tensor, actual_length


# Generate batch from rows
def generate_batch(name, examples, program, length, batch_size=BATCH_SIZE, num_threads=8, shuffle_batch=True):

    # Shuffle batch randomly
    if shuffle_batch:

        # Construct batch from queue of records
        name_batch, examples_batch, program_batch, length_batch = tf.train.shuffle_batch(
            [name, examples, program, length],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=TOTAL_EXAMPLES,
            min_after_dequeue=EPOCH_SIZE)


    # Preserve order of batch
    else:

        # Construct batch from queue of records
        name_batch, examples_batch, program_batch, length_batch = tf.train.batch(
            [name, examples, program, length],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=TOTAL_EXAMPLES)

    return name_batch, examples_batch, program_batch, length_batch


# Generate single training batch of python programs
def get_training_batch():

    # A queue to generate batches
    filename_queue = tf.train.string_input_producer(DATASET_FILENAMES)


    # Decode from string to floating point
    name, examples, program, length = decode_record(filename_queue)


    # Combine example queue into batch
    name_batch, examples_batch, program_batch, length_batch = generate_batch(name, examples, program, length)

    return name_batch, examples_batch, program_batch, length_batch


# Creates a mutated program batch
def mutate_program_batch(program_batch):

    # Generate mutation indices for each character tensor
    batch_mutations = tf.random_uniform(
        [BATCH_SIZE, TOTAL_MUTATIONS], 
        minval=10, 
        maxval=VOCAB_SIZE, 
        dtype=tf.int32)


    # Enumerate each program in batch for mutations
    program_indices = tf.constant([[[i] for _ in range(TOTAL_MUTATIONS)] for i in range(BATCH_SIZE)])


    # Generate mutation indices for each program
    mutation_indices = tf.random_uniform(
        [BATCH_SIZE, TOTAL_MUTATIONS, 1], 
        minval=0, 
        maxval=DATASET_MAXIMUM, 
        dtype=tf.int32)


    # Combine mutation indices with program enumeration
    indices = tf.concat([program_indices, mutation_indices], 2)


    # Convert chartacter mutations to one hot tensor
    updates = tf.one_hot(batch_mutations, VOCAB_SIZE, dtype=tf.float32)


    # Copy program_batch as mutated_batch
    mutated_batch = tf.Variable(tf.zeros(program_batch.shape))
    mutated_op = tf.assign(mutated_batch, program_batch)


    # Perform a random mutation
    mutated_result = tf.scatter_nd_update(mutated_op, indices, updates)

    return mutated_result


# Initialize trainable parameters
def initialize_weights_cpu(name, shape, standard_deviation=0.01, decay_factor=None, collection=None):

    # Force usage of cpu
    with tf.device("/cpu:0"):

        # Sample weights from normal distribution
        weights = tf.get_variable(
            name,
            shape,
            initializer=tf.truncated_normal_initializer(
                stddev=standard_deviation,
                dtype=tf.float32),
            dtype=tf.float32)

    # Add weight decay to loss function
    if decay_factor is not None and collection is not None:

        # Calculate decay with l2 loss
        weight_decay = tf.multiply(
            tf.nn.l2_loss(weights),
            decay_factor,
            name=(name + EXTENSION_LOSS))
        tf.add_to_collection(collection, weight_decay)

    return weights


# Initialize trainable parameters
def initialize_biases_cpu(name, shape):

    # Force usage of cpu
    with tf.device("/cpu:0"):

        # Sample weights from normal distribution
        biases = tf.get_variable(
            name,
            shape,
            initializer=tf.constant_initializer(1.0),
            dtype=tf.float32)

    return biases


# Compute syntax label with brnn
def inference_syntax(program_batch, length_batch):

    # Initialization flag for lstm
    global SYNTAX_INITIALIZED


    # First bidirectional lstm layer
    with tf.variable_scope((PREFIX_SYNTAX + EXTENSION_NUMBER(1)), reuse=SYNTAX_INITIALIZED) as scope:

        # Define forward and backward rnn layers
        lstm_forward = tf.contrib.rnn.LSTMCell(LSTM_SIZE)
        lstm_backward = tf.contrib.rnn.LSTMCell(LSTM_SIZE)


        # Initial state for lstm cell
        initial_state_fw = lstm_forward.zero_state(BATCH_SIZE, tf.float32)
        initial_state_bw = lstm_backward.zero_state(BATCH_SIZE, tf.float32)


        # Compute dropout probabilities
        if USE_DROPOUT:
            lstm_forward = tf.contrib.rnn.DropoutWrapper(
                lstm_forward, 
                input_keep_prob=1.0, 
                output_keep_prob=DROPOUT_PROBABILITY)
            lstm_backward = tf.contrib.rnn.DropoutWrapper(
                lstm_backward, 
                input_keep_prob=1.0, 
                output_keep_prob=DROPOUT_PROBABILITY)


        # Compute rnn activations
        output_batch, state_batch = tf.nn.bidirectional_dynamic_rnn(
            lstm_forward,
            lstm_backward,
            program_batch,
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            sequence_length=length_batch,
            dtype=tf.float32)


        # Add parameters to collection for training
        parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , scope=scope.name)
        tf.add_to_collection((PREFIX_SYNTAX + COLLECTION_PARAMETERS), parameters)


    # Second bidirectional lstm layer
    with tf.variable_scope((PREFIX_SYNTAX + EXTENSION_NUMBER(2)), reuse=SYNTAX_INITIALIZED) as scope:

        # Define forward and backward rnn layers
        lstm_forward = tf.contrib.rnn.LSTMCell(LSTM_SIZE)
        lstm_backward = tf.contrib.rnn.LSTMCell(LSTM_SIZE)


        # Initial state for lstm cell
        initial_state_fw = lstm_forward.zero_state(BATCH_SIZE, tf.float32)
        initial_state_bw = lstm_backward.zero_state(BATCH_SIZE, tf.float32)


        # Compute dropout probabilities
        if USE_DROPOUT:
            lstm_forward = tf.contrib.rnn.DropoutWrapper(
                lstm_forward, 
                input_keep_prob=DROPOUT_PROBABILITY, 
                output_keep_prob=DROPOUT_PROBABILITY)
            lstm_backward = tf.contrib.rnn.DropoutWrapper(
                lstm_backward, 
                input_keep_prob=DROPOUT_PROBABILITY, 
                output_keep_prob=DROPOUT_PROBABILITY)


        # Compute rnn activations
        output_batch, state_batch = tf.nn.bidirectional_dynamic_rnn(
            lstm_forward,
            lstm_backward,
            tf.concat(output_batch, 2),
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            sequence_length=length_batch,
            dtype=tf.float32)


        # Add parameters to collection for training
        parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , scope=scope.name)
        tf.add_to_collection((PREFIX_SYNTAX + COLLECTION_PARAMETERS), parameters)


    # Third attentional dense layer
    with tf.variable_scope((PREFIX_SYNTAX + EXTENSION_NUMBER(3)), reuse=SYNTAX_INITIALIZED) as scope:

        # Take linear combination of hidden states and produce syntax label
        linear_w = initialize_weights_cpu(
            (scope.name + EXTENSION_WEIGHTS), 
            [DATASET_MAXIMUM, LSTM_SIZE*2])
        linear_b = initialize_biases_cpu(
            (scope.name + EXTENSION_BIASES), 
            [1])
        output_batch = tf.add(tf.tensordot(tf.concat(output_batch, 2), linear_w, 2), linear_b)


        # Add parameters to collection for training
        parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , scope=scope.name)
        tf.add_to_collection((PREFIX_SYNTAX + COLLECTION_PARAMETERS), parameters)

    
    # LSTM has been computed at least once, return calculation node
    SYNTAX_INITIALIZED = True
    return tf.reshape(output_batch, [BATCH_SIZE])


# Compute behavior function with brnn
def inference_behavior(program_batch, length_batch):

    # Initialization flag for lstm
    global BEHAVIOR_INITIALIZED


    # First bidirectional lstm layer
    with tf.variable_scope((PREFIX_BEHAVIOR + EXTENSION_NUMBER(1)), reuse=BEHAVIOR_INITIALIZED) as scope:

        # Define forward and backward rnn layers
        lstm_forward = tf.contrib.rnn.LSTMCell(LSTM_SIZE)
        lstm_backward = tf.contrib.rnn.LSTMCell(LSTM_SIZE)


        # Initial state for lstm cell
        initial_state_fw = lstm_forward.zero_state(BATCH_SIZE, tf.float32)
        initial_state_bw = lstm_backward.zero_state(BATCH_SIZE, tf.float32)


        # Compute dropout probabilities
        if USE_DROPOUT:
            lstm_forward = tf.contrib.rnn.DropoutWrapper(
                lstm_forward, 
                input_keep_prob=1.0, 
                output_keep_prob=DROPOUT_PROBABILITY)
            lstm_backward = tf.contrib.rnn.DropoutWrapper(
                lstm_backward, 
                input_keep_prob=1.0, 
                output_keep_prob=DROPOUT_PROBABILITY)


        # Compute rnn activations
        output_batch, state_batch = tf.nn.bidirectional_dynamic_rnn(
            lstm_forward,
            lstm_backward,
            program_batch,
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            sequence_length=length_batch,
            dtype=tf.float32)


        # Add parameters to collection for training
        parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , scope=scope.name)
        tf.add_to_collection((PREFIX_BEHAVIOR + COLLECTION_PARAMETERS), parameters)


    # Second bidirectional lstm layer
    with tf.variable_scope((PREFIX_BEHAVIOR + EXTENSION_NUMBER(2)), reuse=BEHAVIOR_INITIALIZED) as scope:

        # Define forward and backward rnn layers
        lstm_forward = tf.contrib.rnn.LSTMCell(LSTM_SIZE)
        lstm_backward = tf.contrib.rnn.LSTMCell(LSTM_SIZE)


        # Initial state for lstm cell
        initial_state_fw = lstm_forward.zero_state(BATCH_SIZE, tf.float32)
        initial_state_bw = lstm_backward.zero_state(BATCH_SIZE, tf.float32)


        # Compute dropout probabilities
        if USE_DROPOUT:
            lstm_forward = tf.contrib.rnn.DropoutWrapper(
                lstm_forward, 
                input_keep_prob=DROPOUT_PROBABILITY, 
                output_keep_prob=DROPOUT_PROBABILITY)
            lstm_backward = tf.contrib.rnn.DropoutWrapper(
                lstm_backward, 
                input_keep_prob=DROPOUT_PROBABILITY, 
                output_keep_prob=DROPOUT_PROBABILITY)


        # Compute rnn activations
        output_batch, state_batch = tf.nn.bidirectional_dynamic_rnn(
            lstm_forward,
            lstm_backward,
            tf.concat(output_batch, 2),
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            sequence_length=length_batch,
            dtype=tf.float32)


        # Add parameters to collection for training
        parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , scope=scope.name)
        tf.add_to_collection((PREFIX_BEHAVIOR + COLLECTION_PARAMETERS), parameters)


    # Third attentional dense layer
    with tf.variable_scope((PREFIX_BEHAVIOR + EXTENSION_NUMBER(3)), reuse=BEHAVIOR_INITIALIZED) as scope:

        # Compute the weights and biases for behavior hypernetwork
        hypernet_w = []
        hypernet_b = []


        # Compute each layer weights for hypernet
        for i, layer in enumerate(BEHAVIOR_TOPOLOGY):

            # Take linear combination of hidden states and produce hypernet weights
            linear_w_w = initialize_weights_cpu(
                (scope.name + EXTENSION_WEIGHTS + EXTENSION_NUMBER(i) + EXTENSION_WEIGHTS), 
                ([DATASET_MAXIMUM, (LSTM_SIZE * 2)] + layer))
            linear_w_b = initialize_biases_cpu(
                (scope.name + EXTENSION_WEIGHTS + EXTENSION_NUMBER(i) + EXTENSION_BIASES), 
                layer)
            weights = tf.add(tf.tensordot(tf.concat(output_batch, 2), linear_w_w, 2), linear_w_b)


            # Append weights to layers of hypernet
            hypernet_w.append(weights)


            # Take linear combination of hidden states and produce hypernet biases
            linear_b_w = initialize_weights_cpu(
                (scope.name + EXTENSION_BIASES + EXTENSION_NUMBER(i) + EXTENSION_WEIGHTS), 
                ([DATASET_MAXIMUM, (LSTM_SIZE * 2), layer[-1]]))
            linear_b_b = initialize_biases_cpu(
                (scope.name + EXTENSION_BIASES + EXTENSION_NUMBER(i) + EXTENSION_BIASES), 
                [layer[-1]])
            biases = tf.add(tf.tensordot(tf.concat(output_batch, 2), linear_b_w, 2), linear_b_b)


            # Append biases to layers of hypernet
            hypernet_b.append(biases)


            # Add parameters to collection for training
            parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , scope=scope.name)
            tf.add_to_collection((PREFIX_BEHAVIOR + COLLECTION_PARAMETERS), parameters)


    # Compute expected output given input example
    def behavior_function(input_examples):

        # Add additional dimension end of input [64, 10, 1]
        activation = tf.expand_dims(input_examples, -1)


        # Compute hypernet behavior function
        for i in range(BEHAVIOR_DEPTH + 2):
            
            # Reshape previous activation to align elementwise multiplication
            activation = tf.tile(tf.expand_dims(activation, -1), [1, 1, 1, BEHAVIOR_TOPOLOGY[i][-1]])


            # Reshape weights to align elementwise multipliction
            weights = tf.tile(tf.expand_dims(hypernet_w[i], 1), [1, DATASET_IO_EXAMPLES, 1, 1])


            # Reshape biases to align elementwise addition
            biases = tf.tile(tf.expand_dims(hypernet_b[i], 1), [1, DATASET_IO_EXAMPLES, 1])


            # Compute batch-example wise tensor product and relu activation, dead relu offset
            activation = tf.nn.relu(tf.reduce_sum(activation * weights, axis=2) + biases + 10.0)

        return tf.reshape(activation, [BATCH_SIZE, DATASET_IO_EXAMPLES])


    # LSTM has been computed at least once
    BEHAVIOR_INITIALIZED = True
    return behavior_function


# Compute generated program with brnn
def inference_generator(program_batch, length_batch):

    # Initialization flag for lstm
    global GENERATOR_INITIALIZED


    # First bidirectional lstm layer
    with tf.variable_scope((PREFIX_GENERATOR + EXTENSION_NUMBER(1)), reuse=GENERATOR_INITIALIZED) as scope:

        # Define forward and backward rnn layers
        lstm_forward = tf.contrib.rnn.LSTMCell(LSTM_SIZE)
        lstm_backward = tf.contrib.rnn.LSTMCell(LSTM_SIZE)


        # Initial state for lstm cell
        initial_state_fw = lstm_forward.zero_state(BATCH_SIZE, tf.float32)
        initial_state_bw = lstm_backward.zero_state(BATCH_SIZE, tf.float32)


        # Compute dropout probabilities
        if USE_DROPOUT:
            lstm_forward = tf.contrib.rnn.DropoutWrapper(
                lstm_forward, 
                input_keep_prob=1.0, 
                output_keep_prob=DROPOUT_PROBABILITY)
            lstm_backward = tf.contrib.rnn.DropoutWrapper(
                lstm_backward, 
                input_keep_prob=1.0, 
                output_keep_prob=DROPOUT_PROBABILITY)


        # Compute rnn activations
        output_batch, state_batch = tf.nn.bidirectional_dynamic_rnn(
            lstm_forward,
            lstm_backward,
            program_batch,
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            sequence_length=length_batch,
            dtype=tf.float32)


        # Add parameters to collection for training
        parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , scope=scope.name)
        tf.add_to_collection((PREFIX_GENERATOR + COLLECTION_PARAMETERS), parameters)


    # Second bidirectional lstm layer
    with tf.variable_scope((PREFIX_GENERATOR + EXTENSION_NUMBER(2)), reuse=GENERATOR_INITIALIZED) as scope:

        # Define forward and backward rnn layers
        lstm_forward = tf.contrib.rnn.LSTMCell(LSTM_SIZE)
        lstm_backward = tf.contrib.rnn.LSTMCell(LSTM_SIZE)


        # Initial state for lstm cell
        initial_state_fw = lstm_forward.zero_state(BATCH_SIZE, tf.float32)
        initial_state_bw = lstm_backward.zero_state(BATCH_SIZE, tf.float32)


        # Compute dropout probabilities
        if USE_DROPOUT:
            lstm_forward = tf.contrib.rnn.DropoutWrapper(
                lstm_forward, 
                input_keep_prob=DROPOUT_PROBABILITY, 
                output_keep_prob=DROPOUT_PROBABILITY)
            lstm_backward = tf.contrib.rnn.DropoutWrapper(
                lstm_backward, 
                input_keep_prob=DROPOUT_PROBABILITY, 
                output_keep_prob=DROPOUT_PROBABILITY)


        # Compute rnn activations
        output_batch, state_batch = tf.nn.bidirectional_dynamic_rnn(
            lstm_forward,
            lstm_backward,
            tf.concat(output_batch, 2),
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            sequence_length=length_batch,
            dtype=tf.float32)


        # Add parameters to collection for training
        parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , scope=scope.name)
        tf.add_to_collection((PREFIX_GENERATOR + COLLECTION_PARAMETERS), parameters)


    # Third attentional dense layer
    with tf.variable_scope((PREFIX_GENERATOR + EXTENSION_NUMBER(3)), reuse=GENERATOR_INITIALIZED) as scope:

        # Take linear combination of hidden states and produce syntax label
        linear_w = initialize_weights_cpu(
            (scope.name + EXTENSION_WEIGHTS), 
            [DATASET_MAXIMUM, LSTM_SIZE*2, DATASET_MAXIMUM, VOCAB_SIZE])
        linear_b = initialize_biases_cpu(
            (scope.name + EXTENSION_BIASES), 
            [DATASET_MAXIMUM, VOCAB_SIZE])
        output_batch = tf.add(tf.tensordot(tf.concat(output_batch, 2), linear_w, 2), linear_b)


        # Add parameters to collection for training
        parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , scope=scope.name)
        tf.add_to_collection((PREFIX_GENERATOR + COLLECTION_PARAMETERS), parameters)

    
    # LSTM has been computed at least once, return calculation node
    GENERATOR_INITIALIZED = True
    return tf.reshape(output_batch, [BATCH_SIZE, DATASET_MAXIMUM, VOCAB_SIZE])


# Utility function reset initialization flags
def reset_kernel():

    # Define global kernal states
    global SYNTAX_INITIALIZED
    global BEHAVIOR_INITIALIZED
    global GENERATOR_INITIALIZED
    global STEP_INCREMENTED


    # Assign states to initial empty
    SYNTAX_INITIALIZED = None
    BEHAVIOR_INITIALIZED = None
    GENERATOR_INITIALIZED = None
    STEP_INCREMENTED = False


# Compute loss for syntax discriminator
def loss(prediction, labels, collection):

    # Calculate huber loss of prediction
    huber_loss = tf.losses.huber_loss(labels, prediction)


    # Calculate the mean loss across batch
    tf.add_to_collection(collection, huber_loss)

    return huber_loss


# Compute loss gradient and update parameters
def train(loss, parameters):
    
    # Initialization flag for global step
    global STEP_INCREMENTED


    # Keep track of current training step
    global_step = tf.train.get_or_create_global_step()


    # Decay learning rate
    learning_rate = tf.train.exponential_decay(
        INITIAL_LEARNING_RATE,
        global_step,
        DECAY_STEPS,
        DECAY_FACTOR,
        staircase=True)


    # Create optimizer for gradient updates
    optimizer = tf.train.AdamOptimizer(learning_rate)


    # Global step has not yet been incremented
    if not STEP_INCREMENTED:

        # Minimize loss with respect to parameters, and increment step
        gradient = optimizer.minimize(
            loss, 
            var_list=parameters,
            global_step=global_step)


    # Global step has already been incremented
    else:

        # Minimize loss with respect to parameters
        gradient = optimizer.minimize(
            loss, 
            var_list=parameters)


    # Step has been decremented at least once
    STEP_INCREMENTED = True
    return gradient


# Run single training cycle on dataset
def train_epf_5(num_epochs=1):

    # Reset lstm kernel
    reset_kernel()


    # Convert epoch to batch steps
    num_steps = num_epochs * EPOCH_SIZE


    # Create new graph
    with tf.Graph().as_default():

        # Compute single training batch
        name_batch, examples_batch, program_batch, length_batch = get_training_batch()


        # Obtain character mutations of programs
        mutated_batch = mutate_program_batch(program_batch)


        # Compute syntax of original source code
        syntax_batch = inference_syntax(program_batch, length_batch)
        syntax_loss = loss(
            syntax_batch, 
            tf.constant([THRESHOLD_UPPER for i in range(BATCH_SIZE)], tf.float32),
            (PREFIX_SYNTAX + COLLECTION_LOSSES))


        # Compute syntax of mutated source code
        mutated_syntax_batch = inference_syntax(mutated_batch, length_batch)
        mutated_syntax_loss = loss(
            mutated_syntax_batch, 
            tf.constant([THRESHOLD_LOWER for i in range(BATCH_SIZE)], tf.float32),
            (PREFIX_SYNTAX + COLLECTION_LOSSES))


        # Slice examples into input output
        input_examples_batch = tf.strided_slice(
            examples_batch, 
            [0, 0], 
            [BATCH_SIZE, (DATASET_IO_EXAMPLES * 2)], 
            strides=[1, 2])
        output_examples_batch = tf.strided_slice(
            examples_batch, 
            [0, 1], 
            [BATCH_SIZE, (DATASET_IO_EXAMPLES * 2)], 
            strides=[1, 2])


        # Compute behavior function of original code
        behavior_batch = inference_behavior(program_batch, length_batch)
        behavior_prediction = behavior_batch(input_examples_batch)
        behavior_loss = loss(
            behavior_prediction, 
            output_examples_batch,
            (PREFIX_BEHAVIOR + COLLECTION_LOSSES))


        # Compute behavior function of mutated code
        mutated_behavior_batch = inference_behavior(mutated_batch, length_batch)
        mutated_behavior_prediction = mutated_behavior_batch(input_examples_batch)
        mutated_behavior_loss = loss(
            mutated_behavior_prediction, 
            output_examples_batch,
            (PREFIX_BEHAVIOR + COLLECTION_LOSSES))


        # Calculate the corrected code with generator
        generated_batch = inference_generator(program_batch, length_batch)
        mutated_generated_batch = inference_generator(mutated_batch, length_batch)


        # Compute syntax classification of generated program
        syntax_generated_batch = inference_syntax(generated_batch, length_batch)
        syntax_generated_loss = loss(
            syntax_generated_batch, 
            tf.constant([THRESHOLD_UPPER for i in range(BATCH_SIZE)], tf.float32),
            (PREFIX_GENERATOR + COLLECTION_LOSSES))


        # Compute syntax classification of generated program from mutated input
        syntax_mutated_generated_batch = inference_syntax(mutated_generated_batch, length_batch)
        syntax_mutated_generated_loss = loss(
            syntax_mutated_generated_batch, 
            tf.constant([THRESHOLD_UPPER for i in range(BATCH_SIZE)], tf.float32),
            (PREFIX_GENERATOR + COLLECTION_LOSSES))


        # Compute the behavior of generated program
        behavior_generated_batch = inference_behavior(generated_batch, length_batch)
        behavior_generated_prediction = behavior_generated_batch(input_examples_batch)
        behavior_generated_loss = loss(
            behavior_generated_prediction, 
            behavior_prediction,
            (PREFIX_GENERATOR + COLLECTION_LOSSES))


        # Compute the behavior of generated program from mutated input
        behavior_mutated_generated_batch = inference_behavior(mutated_generated_batch, length_batch)
        behavior_mutated_generated_prediction = behavior_mutated_generated_batch(input_examples_batch)
        behavior_mutated_generated_loss = loss(
            behavior_mutated_generated_prediction, 
            mutated_behavior_prediction,
            (PREFIX_GENERATOR + COLLECTION_LOSSES))


        # Obtain parameters for syntax and behavior discriminator networks
        syntax_parameters = tf.get_collection(PREFIX_SYNTAX + COLLECTION_PARAMETERS)
        behavior_parameters = tf.get_collection(PREFIX_BEHAVIOR + COLLECTION_PARAMETERS)
        generator_parameters = tf.get_collection(PREFIX_GENERATOR + COLLECTION_PARAMETERS)


        # Obtain loss for which to minimize with respect to parameters
        syntax_loss = tf.add_n(tf.get_collection(PREFIX_SYNTAX + COLLECTION_LOSSES))
        behavior_loss = tf.add_n(tf.get_collection(PREFIX_BEHAVIOR + COLLECTION_LOSSES))
        generator_loss = tf.add_n(tf.get_collection(PREFIX_GENERATOR + COLLECTION_LOSSES))
        total_loss = syntax_loss + behavior_loss + generator_loss


        # Calculate gradient for each set of parameters
        syntax_gradient = train(syntax_loss, syntax_parameters)
        behavior_gradient = train(behavior_loss, behavior_parameters)
        generator_gradient = train(generator_loss, generator_parameters)
        gradient_batch = tf.group(syntax_gradient, behavior_gradient)


        # Report testing progress
        class DataSaver(tf.train.SessionRunHook):

            # Session is initialized
            def begin(self):
                self.start_time = time()
                self.loss_points = []
                self.iteration_points = []


            # Just before inference
            def before_run(self, run_context):
                return tf.train.SessionRunArgs([
                    tf.train.get_global_step(),
                    total_loss])


            # Just after inference
            def after_run(self, run_context, run_values):
                
                # Obtain graph results
                current_step, loss_value = run_values.results


                # Calculate weighted speed
                current_time = time()
                batch_speed = 1.0 / (current_time - self.start_time + 1e-3)
                self.start_time = current_time


                # Update every period of steps
                if current_step % max(num_steps // TOTAL_LOGS, 1) == 0:


                    # Display date, batch speed, estimated time, loss, and accuracy
                    print(
                        datetime.now(),
                        "CUR: %d" % current_step,
                        "REM: %d" % (num_steps - current_step),
                        "SPD: %.2f bat/sec" % batch_speed,
                        "ETA: %.2f hrs" % ((num_steps - current_step) / batch_speed / 60 / 60),
                        "LOSS: %.2f" % loss_value)


                    # Record current loss
                    self.loss_points.append(loss_value)
                    self.iteration_points.append(current_step)


        # Prepare to save and load models
        model_saver = tf.train.Saver()


        # Track datapoints as testing progresses
        data_saver = DataSaver()


        # Perform computation cycle based on graph
        with tf.train.MonitoredTrainingSession(hooks=[
            tf.train.StopAtStepHook(num_steps=num_steps),
            tf.train.CheckpointSaverHook(
                CHECKPOINT_BASEDIR,
                save_steps=EPOCH_SIZE,
                saver=model_saver),
            data_saver]) as session:

            # Repeat training iteratively
            while not session.should_stop():

                # Run single batch of training
                session.run(gradient_batch)


    # Construct and save training loss plot
    plt.plot(
        data_saver.iteration_points, 
        data_saver.loss_points,
        "b--o")
    plt.xlabel("Batch Iteration")
    plt.ylabel("Mean Huber Loss")
    plt.yscale("log")
    plt.savefig(
        PLOT_BASEDIR + 
        datetime.now().strftime("%Y_%B_%d_%H_%M_%S") + 
        "_training_loss.png")
    plt.close()


# Run single testing cycle on dataset
def test_epf_5(model_checkpoint):

    # Reset lstm kernel
    reset_kernel()


    # Create new graph
    with tf.Graph().as_default():

        # Compute single training batch
        name_batch, examples_batch, program_batch, length_batch = get_training_batch()


        # Obtain character mutations of programs
        mutated_batch = mutate_program_batch(program_batch)


        # Compute syntax of corrected code
        syntax_batch = inference_syntax(program_batch, length_batch)
        mutated_syntax_batch = inference_syntax(mutated_batch, length_batch)


        # Slice examples into input output
        input_examples_batch = tf.strided_slice(
            examples_batch, 
            [0, 0], 
            [BATCH_SIZE, (DATASET_IO_EXAMPLES * 2)], 
            strides=[1, 2])
        output_examples_batch = tf.strided_slice(
            examples_batch, 
            [0, 1], 
            [BATCH_SIZE, (DATASET_IO_EXAMPLES * 2)], 
            strides=[1, 2])


        # Compute behavior function of original code
        behavior_batch = inference_behavior(program_batch, length_batch)
        behavior_prediction = behavior_batch(input_examples_batch)
        behavior_error = behavior_prediction - output_examples_batch


        # Compute behavior function of mutated code
        mutated_behavior_batch = inference_behavior(mutated_batch, length_batch)
        mutated_behavior_prediction = mutated_behavior_batch(input_examples_batch)
        mutated_behavior_error = mutated_behavior_prediction - output_examples_batch


        # Calculate the corrected code with generator
        generated_batch = inference_generator(program_batch, length_batch)
        generated_string = detokenize_program(generated_batch)


        # Calculate the corrected code with generator from mutated input
        mutated_generated_batch = inference_generator(mutated_batch, length_batch)
        mutated_generated_string = detokenize_program(mutated_generated_batch)


        # Group the previous operations
        group_batch = tf.group(
            syntax_batch, 
            mutated_syntax_batch,
            behavior_error,
            mutated_behavior_error,
            generated_string,
            mutated_generated_string)


        # Binary classification decision boundary for precision recall
        SYNTAX_THRESHOLD = THRESHOLD_LOWER


        # Report testing progress
        class DataSaver(tf.train.SessionRunHook):

            
            # Session is initialized
            def begin(self):
                self.syntax_accuracy = 0.0
                self.error_mean = np.zeros((DATASET_IO_EXAMPLES))
                self.error_std = np.zeros((DATASET_IO_EXAMPLES))
                self.precision_points = []
                self.recall_points = []
                self.generated_programs = []


            # Just before inference
            def before_run(self, run_context):
                return tf.train.SessionRunArgs([
                    syntax_batch, 
                    mutated_syntax_batch,
                    behavior_error,
                    mutated_behavior_error,
                    generated_string,
                    mutated_generated_string])


            # Just after inference
            def after_run(self, run_context, run_values):

                # Increment threshold after calculation
                nonlocal SYNTAX_THRESHOLD


                # Obtain graph results
                syntax_value, mutated_syntax_value, behavior_value, mutated_behavior_value, generated_value, mutated_generated_value = run_values.results


                # Calculations for precision and recall
                false_positives = sum([1. for element in mutated_syntax_value if element >= SYNTAX_THRESHOLD])
                false_negatives = sum([1. for element in syntax_value if element <= SYNTAX_THRESHOLD])


                # Calculate precision metric true_positive/(true_positive + false_positive)
                precision = BATCH_SIZE / (BATCH_SIZE + false_positives)

                
                # Calculate precision metric true_positive/(true_positive + false_positive)
                recall = BATCH_SIZE / (BATCH_SIZE + false_negatives)


                # Calculate the accuracy at this decision threshold
                current_syntax_accuracy = (BATCH_SIZE * 2 - false_positives - false_negatives) / (BATCH_SIZE * 2)
                self.syntax_accuracy += SAMPLE_WEIGHT(SYNTAX_THRESHOLD) * current_syntax_accuracy

                
                # Record current precision recall
                self.precision_points.append(precision)
                self.recall_points.append(recall)


                # Calculate mean and standard deviation of behavior error
                self.error_mean += behavior_value.mean(axis=0) / THRESHOLD_RANGE 
                self.error_mean += mutated_behavior_value.mean(axis=0) / THRESHOLD_RANGE 
                self.error_std += behavior_value.std(axis=0) / THRESHOLD_RANGE
                self.error_std += mutated_behavior_value.std(axis=0) / THRESHOLD_RANGE


                # Decode program strings from byte matrix
                generated_value_string = [
                    "".join(map(
                        (lambda p: p.decode("utf-8")), 
                        program)) for program in generated_value.tolist()]
                mutated_generated_value_string = [
                    "".join(map(
                        (lambda p: p.decode("utf-8")), 
                        program)) for program in mutated_generated_value.tolist()]
                self.generated_programs += generated_value_string + mutated_generated_value_string


                # Print result for verification
                print(
                    datetime.now(),
                    "THD: %.2f" % SYNTAX_THRESHOLD,
                    "PRE: %.2f" % precision, 
                    "REC: %.2f" % recall,
                    "ACC: %.2f" % (self.syntax_accuracy * 100),
                    "ERR Mean: %.2f" % self.error_mean.mean(),
                    "ERR STD: %.2f" % self.error_std.mean())

                
                # Calculate new decision threshold
                SYNTAX_THRESHOLD += THRESHOLD_DELTA


        # Prepare to save and load models
        model_saver = tf.train.Saver()


        # Track datapoints as testing progresses
        data_saver = DataSaver()


        # Perform computation cycle based on graph
        with tf.train.MonitoredTrainingSession(hooks=[
            data_saver]) as session:

            # Load progress from checkpoint
            model_saver.restore(session, model_checkpoint)


            # Repeat testing iteratively for varying decision thresholds
            for i in range(THRESHOLD_RANGE):

                # Run single batch of testing
                session.run(group_batch)


    # Construct and precision recall save plot
    plt.plot(
        data_saver.recall_points, 
        data_saver.precision_points,
        "b--o")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(
        PLOT_BASEDIR + 
        datetime.now().strftime("%Y_%B_%d_%H_%M_%S") + 
        "_precision_recall.png")
    plt.close()


    # Construct and save behavior error plot
    plt.errorbar(
        np.arange(10), 
        data_saver.error_mean, 
        yerr=data_saver.error_std, 
        fmt="b--o", 
        ecolor="g", 
        capsize=10)
    plt.xlabel("IO Example")
    plt.ylabel("Behavior Error")
    plt.savefig(
        PLOT_BASEDIR + 
        datetime.now().strftime("%Y_%B_%d_%H_%M_%S") + 
        "_behavior_error.png")
    plt.close()


    # Write file containing generated programs
    with open(
        PLOT_BASEDIR + 
        datetime.now().strftime("%Y_%B_%d_%H_%M_%S") + 
        "_generated_programs.txt", "w") as output_file:

        # Write each program string to separate line
        for line in data_saver.generated_programs:
            output_file.write(line)
