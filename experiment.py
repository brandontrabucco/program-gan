import tensorflow as tf
import string as sn
import numpy as np


# The location on the disk of project
PROJECT_BASEDIR = ("C:/Users/brand/Google Drive/Academic/Research/" +
    "Program Synthesis with Deep Learning/Repo/program-gan/")


# The location on the disk of checkpoints
CHECKPOINT_BASEDIR = (PROJECT_BASEDIR + "Checkpoints/")


# The location on the disk of project
DATASET_BASEDIR = ("C:/Users/brand/Google Drive/Academic/Research/" +
    "Program Synthesis with Deep Learning/Datasets/")


# Filenames associated with program dataset
DATASET_FILENAMES_PYTHON = [
    (DATASET_BASEDIR + "epf_5_dataset.csv")
]


# Locate dataset files on hard disk
for FILE_PYTHON in DATASET_FILENAMES_PYTHON:
    if not tf.gfile.Exists(FILE_PYTHON):
        raise ValueError('Failed to find file: ' + FILE_PYTHON)


# Dataset configuration constants
DATASET_IO_EXAMPLES = 10
DATASET_COLUMNS = (DATASET_IO_EXAMPLES * 2) + 2
DATASET_DEFAULT = "0"


# Tokenization parameters for python
DATASET_VOCABULARY = sn.printable
DATASET_MAXIMUM = 64


# Convert elements of python source code to one-hot token vectors
def tokenize_source_code_python(source_code_python, vocabulary=DATASET_VOCABULARY):

    # List allowed characters
    mapping_characters = tf.string_split([vocabulary], delimiter="")


    # List characters in each word
    input_characters = tf.string_split([source_code_python], delimiter="")


    # Create integer lookup table
    lookup_table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_characters.values, default_value=0)


    # Query lookup table
    one_hot_tensor = tf.one_hot(lookup_table.lookup(input_characters.values), len(vocabulary), dtype=tf.float32)


    # Calculate actual sequence length
    actual_length = tf.size(one_hot_tensor) // len(vocabulary)


    # Pad input to match DATASET_MAXIMUM
    expanded_tensor = tf.pad(one_hot_tensor, [[0, (DATASET_MAXIMUM - actual_length)], [0, 0]])

    return tf.reshape(expanded_tensor, [DATASET_MAXIMUM, len(vocabulary)]), actual_length


def detokenize_source_code_python(one_hot_tensor, vocabulary=DATASET_VOCABULARY):

    # List allowed characters
    mapping_characters = tf.string_split([vocabulary], delimiter="")


    # Select one hot index
    indices = tf.argmax(one_hot_tensor, axis=2)


    # Create integer lookup table
    lookup_table = tf.contrib.lookup.index_to_string_table_from_tensor(mapping_characters.values, default_value="UNKNOWN")


    # Lookup corrosponding string
    program = lookup_table.lookup(indices)

    return program


# Read single row words
def decode_record_python(filename_queue, num_columns=DATASET_COLUMNS, default_value=DATASET_DEFAULT):

    # Attach text file reader
    DATASET_READER = tf.TextLineReader(skip_header_lines=1)


    # Read single line from dataset
    key, value_text = DATASET_READER.read(filename_queue)


    # Decode line to columns of strings
    name_column, *example_columns, function_column = tf.decode_csv(value_text, [[default_value] for i in range(num_columns)])


    # Convert IO examples from string to float32
    example_columns = tf.string_to_number(tf.stack(example_columns), out_type=tf.float32)


    # Convert python code to tokenized one-hot vectors
    program_tensor, actual_length = tokenize_source_code_python(function_column)

    return name_column, example_columns, program_tensor, actual_length


# Batch configuration constants
BATCH_SIZE = 64
NUM_THREADS = 4
TOTAL_EXAMPLES = 9330
EPOCH_SIZE = TOTAL_EXAMPLES // BATCH_SIZE


# Generate batch from rows
def generate_batch(name, examples, program, length, batch_size=BATCH_SIZE, num_threads=NUM_THREADS, shuffle_batch=True):

    # Shuffle batch randomly
    if shuffle_batch:

        # Construct batch from queue of records
        name_batch, examples_batch, program_batch, length_batch = tf.train.shuffle_batch(
            [name, examples, program, length],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=TOTAL_EXAMPLES,
            min_after_dequeue=(TOTAL_EXAMPLES // 1000))


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
def training_batch_python():

    # A queue to generate batches
    filename_queue = tf.train.string_input_producer(DATASET_FILENAMES_PYTHON)


    # Decode from string to floating point
    name, examples, program, length = decode_record_python(filename_queue)


    # Combine example queue into batch
    name_batch, examples_batch, program_batch, length_batch = generate_batch(name, examples, program, length)

    return name_batch, examples_batch, program_batch, length_batch


# Mutation parameter
NO_OF_MUTATIONS = 10
VOCAB_SIZE = len(DATASET_VOCABULARY)


# Creates a mutated program batch
def get_mutated_batch(program_batch):

    # generates batch_mutations a 2D tensor of integers with shape (BATCH_SIZE, NO_OF_MUTATIONS) 
    # that represents the mutations to the batch
    batch_mutations = tf.random_uniform(
        [BATCH_SIZE, NO_OF_MUTATIONS], 
        minval=10, 
        maxval=VOCAB_SIZE, 
        dtype=tf.int32)


    # generates a tensor with shape [BATCH_SIZE, NO_OF_MUTATIONS, 1]. BATCH_SIZE=2 
    # and NO_OF_MUTATIONS=3 yields [[[0], [0], [0]], [[1], [1], [1]]].
    # The numbers represent the index of the program to be mutated within the batch
    # so 0 refers to the first program in batch.
    program_indices = tf.constant([[[i] for _ in range(NO_OF_MUTATIONS)] for i in range(BATCH_SIZE)])


    # generates a tensor with shape [BATCH_SIZE, NO_OF_MUTATIONS, 1]. Each number represents 
    # a random index within the program at which mutation occurs.
    # the indices may repeat, so actual no of mutations within the program <= NO_OF_MUTATIONS
    mutation_indices = tf.random_uniform(
        [BATCH_SIZE, NO_OF_MUTATIONS, 1], 
        minval=0, 
        maxval=DATASET_MAXIMUM, 
        dtype=tf.int32)


    # Generates the locations within batch at which the batch_mutations occur. 
    # Each index is [index of program within batch, index within program]
    indices = tf.concat([program_indices, mutation_indices], 2)


    # creates one hot tensor with random one hot vecs
    updates = tf.one_hot(batch_mutations, VOCAB_SIZE, dtype=tf.float32)


    # make a copy of program_batch as mutated_batch
    mutated_batch = tf.Variable(tf.zeros(program_batch.shape))
    mutated_op = tf.assign(mutated_batch, program_batch)


    # mutates mutated_batch
    mutated_result = tf.scatter_nd_update(mutated_op, indices, updates)

    return mutated_result


# Prefix model nomenclature
PREFIX_RNN = "rnn"
PREFIX_DENSE = "dense"
PREFIX_SOFTMAX = "softmax"
PREFIX_TOTAL = "total"
PREFIX_GENERATOR = "generator"
PREFIX_SYNTAX = "syntax"
PREFIX_BEHAVIOR = "behavior"


# Extension model nomenclature
EXTENSION_NUMBER = (lambda number: "_" + str(number))
EXTENSION_LOSS = "_loss"
EXTENSION_WEIGHTS = "_weights"
EXTENSION_BIASES = "_biases"
EXTENSION_OFFSET = "_offset"
EXTENSION_SCALE = "_scale"
EXTENSION_ACTIVATION = "_activation"
EXTENSION_COLUMN = "_column"


# Collection model nomenclature
COLLECTION_LOSSES = "losses"
COLLECTION_PARAMETERS = "parameters"
COLLECTION_ACTIVATIONS = "activations"


# Initialize trainable parameters
def initialize_weights_cpu(name, shape, standard_deviation=0.01, decay_factor=None):

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
    if decay_factor is not None:

        # Calculate decay with l2 loss
        weight_decay = tf.multiply(
            tf.nn.l2_loss(weights),
            decay_factor,
            name=(name + EXTENSION_LOSS))
        tf.add_to_collection(COLLECTION_LOSSES, weight_decay)

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


# Compute corrected tokenized code with rnn
def inference_generator_python(program_batch):

    # Placeholder weights for computation check
    generator_weights = initialize_weights_cpu((PREFIX_GENERATOR + EXTENSION_WEIGHTS), program_batch.shape)

    return program_batch * generator_weights


# Compute behavior function with rnn
def inference_behavior_python(program_batch):

    # Placeholder weights for computation check
    behavior_weights = initialize_weights_cpu((PREFIX_BEHAVIOR + EXTENSION_WEIGHTS), [BATCH_SIZE])


    # Compute expected output given input example
    def behavior_function(input_example):

        return input_example * behavior_weights

    return behavior_function


# Hidden size of LSTM recurrent cell
LSTM_SIZE = len(DATASET_VOCABULARY) * 2
LSTM_INITIALIZED = None
DROPOUT_PROBABILITY = 0.5


# Compute syntax label with brnn
def inference_syntax_python(program_batch, length_batch):

    # Initialization flag for lstm
    global LSTM_INITIALIZED


    # First bidirectional lstm layer
    with tf.variable_scope((PREFIX_SYNTAX + EXTENSION_NUMBER(1)), reuse=LSTM_INITIALIZED) as scope:

        # Define forward and backward rnn layers
        lstm_forward = tf.contrib.rnn.LSTMCell(LSTM_SIZE)
        lstm_backward = tf.contrib.rnn.LSTMCell(LSTM_SIZE)


        # Initial state for lstm cell
        initial_state_fw = lstm_forward.zero_state(BATCH_SIZE, tf.float32)
        initial_state_bw = lstm_backward.zero_state(BATCH_SIZE, tf.float32)


        # Compute dropout probabilities
        dropout_forward = tf.contrib.rnn.DropoutWrapper(
            lstm_forward, 
            input_keep_prob=DROPOUT_PROBABILITY, 
            output_keep_prob=DROPOUT_PROBABILITY)
        dropout_backward = tf.contrib.rnn.DropoutWrapper(
            lstm_backward, 
            input_keep_prob=DROPOUT_PROBABILITY, 
            output_keep_prob=DROPOUT_PROBABILITY)


        # Compute rnn activations
        output_batch, state_batch = tf.nn.bidirectional_dynamic_rnn(
            dropout_forward,
            dropout_backward,
            program_batch,
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            sequence_length=length_batch,
            dtype=tf.float32)


    # Second bidirectional lstm layer
    with tf.variable_scope((PREFIX_SYNTAX + EXTENSION_NUMBER(2)), reuse=LSTM_INITIALIZED) as scope:

        # Define forward and backward rnn layers
        lstm_forward = tf.contrib.rnn.LSTMCell(LSTM_SIZE)
        lstm_backward = tf.contrib.rnn.LSTMCell(LSTM_SIZE)


        # Initial state for lstm cell
        initial_state_fw = lstm_forward.zero_state(BATCH_SIZE, tf.float32)
        initial_state_bw = lstm_backward.zero_state(BATCH_SIZE, tf.float32)


        # Compute dropout probabilities
        dropout_forward = tf.contrib.rnn.DropoutWrapper(
            lstm_forward, 
            input_keep_prob=DROPOUT_PROBABILITY, 
            output_keep_prob=DROPOUT_PROBABILITY)
        dropout_backward = tf.contrib.rnn.DropoutWrapper(
            lstm_backward, 
            input_keep_prob=DROPOUT_PROBABILITY, 
            output_keep_prob=DROPOUT_PROBABILITY)


        # Compute rnn activations
        output_batch, state_batch = tf.nn.bidirectional_dynamic_rnn(
            dropout_forward,
            dropout_backward,
            tf.concat(output_batch, 2),
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            sequence_length=length_batch,
            dtype=tf.float32)


    # Third attentional dense layer
    with tf.variable_scope((PREFIX_SYNTAX + EXTENSION_NUMBER(3)), reuse=LSTM_INITIALIZED) as scope:

        # Take linear combination of hidden states and produce syntax label
        linear_w = initialize_weights_cpu(
            (scope.name + EXTENSION_WEIGHTS), 
            [DATASET_MAXIMUM, LSTM_SIZE*2])
        linear_b = initialize_biases_cpu((scope.name + EXTENSION_BIASES), [1])
        logits = tf.add(tf.tensordot(tf.concat(output_batch, 2), linear_w, 2), linear_b)

    
    # LSTM has been computed at least once, return calculation node
    LSTM_INITIALIZED = True
    return logits


# Compute loss for syntax discriminator
def loss(prediction, labels):

    # Calculate huber loss of prediction
    huber_loss = tf.losses.huber_loss(labels, prediction)


    # Calculate the mean loss across batch
    huber_loss_mean = tf.reduce_mean(huber_loss)
    tf.add_to_collection(COLLECTION_LOSSES, huber_loss_mean)

    return huber_loss_mean


# Hyperparameters
INITIAL_LEARNING_RATE = 0.0005
DECAY_STEPS = EPOCH_SIZE
DECAY_FACTOR = 0.5


# Compute loss gradient and update parameters
def train(total_loss):

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


    # Minimize loss using optimizer
    gradient = optimizer.minimize(total_loss, global_step=global_step)

    return gradient


# Run single training cycle on dataset
def train_epf_5(num_epoch=1):

    # Reset lstm kernel
    global LSTM_INITIALIZED
    LSTM_INITIALIZED = None


    # Watch compute time per batch
    from time import time
    from datetime import datetime


    # Convert epoch to batch steps
    num_steps = num_epoch * EPOCH_SIZE


    # Create new graph
    with tf.Graph().as_default():

        # Compute single training batch
        name_batch, examples_batch, program_batch, length_batch = training_batch_python()


        # Obtain character mutations of programs
        mutated_batch = get_mutated_batch(program_batch)


        # Compute syntax of original and mutated code
        syntax_batch = inference_syntax_python(program_batch, length_batch)
        mutated_syntax_batch = inference_syntax_python(mutated_batch, length_batch)
        syntax_loss = loss(syntax_batch, tf.constant([1. for i in range(BATCH_SIZE)], tf.float32))
        mutated_syntax_loss = loss(mutated_syntax_batch, tf.constant([0. for i in range(BATCH_SIZE)], tf.float32))


        # Obtain total loss of entire model
        total_loss = tf.add_n(tf.get_collection(COLLECTION_LOSSES), name=(PREFIX_TOTAL + EXTENSION_LOSS))
        gradient_batch = train(total_loss)


        # Store datapoints each epoch for plot
        loss_points = []
        iteration_points = []


        # Report testing progress
        class LogProgressHook(tf.train.SessionRunHook):

            # Session is initialized
            def begin(self):
                self.current_step = 0
                self.batch_speed = 0


            # Just before inference
            def before_run(self, run_context):
                self.current_step += 1
                self.start_time = time()
                return tf.train.SessionRunArgs([syntax_batch, total_loss])


            # Just after inference
            def after_run(self, run_context, run_values):

                # Calculate weighted speed
                self.batch_speed = (0.2 * self.batch_speed) + (0.8 * (1.0 / (time() - self.start_time + 1e-3)))


                # Update every period of steps
                if (self.current_step % 10 == 0):

                    # Obtain graph results
                    syntax_value, loss_value = run_values.results


                    # Display date, batch speed, estimated time, loss, and accuracy
                    print(
                        datetime.now(),
                        "CUR: %d" % self.current_step,
                        "REM: %d" % (num_steps - self.current_step),
                        "SPD: %.2f bat/sec" % self.batch_speed,
                        "ETA: %.2f hrs" % ((num_steps - self.current_step) / self.batch_speed / 60 / 60),
                        "L: %.2f" % loss_value)


                    # Record current loss
                    loss_points.append(loss_value)
                    iteration_points.append(self.current_step)


        # Prepare to save and load models
        model_saver = tf.train.Saver()


        # Perform computation cycle based on graph
        with tf.train.MonitoredTrainingSession(hooks=[
            tf.train.StopAtStepHook(num_steps=num_steps),
            tf.train.CheckpointSaverHook(
                CHECKPOINT_BASEDIR,
                save_steps=EPOCH_SIZE,
                saver=model_saver),
            LogProgressHook()]) as session:

            # Repeat training iteratively
            while not session.should_stop():

                # Run single batch of training
                gradient_value = session.run(gradient_batch)


     # Construct and save plot
    import matplotlib.pyplot as plt
    plt.scatter(iteration_points, loss_points)
    plt.xlabel("Batch Iteration")
    plt.ylabel("Mean Huber Syntax Loss")
    plt.savefig(datetime.now().strftime("%Y_%B_%d_%H_%M_%S") + "_syntax_training_loss.png")
    plt.close()


DECISION_UPPER_BOUND = 1.0
DECISION_LOWER_BOUND = 0.0
DECISION_RANGE = 100


# Run single training cycle on dataset
def test_epf_5(model_checkpoint):

    # Reset lstm kernel
    global LSTM_INITIALIZED
    LSTM_INITIALIZED = None


    # Check current date and time
    from datetime import datetime


    # Create new graph
    with tf.Graph().as_default():

        # Compute single training batch
        name_batch, examples_batch, program_batch, length_batch = training_batch_python()


        # Obtain character mutations of programs
        mutated_batch = get_mutated_batch(program_batch)


        # Compute syntax of corrected code
        syntax_batch = inference_syntax_python(program_batch, length_batch)
        mutated_syntax_batch = inference_syntax_python(mutated_batch, length_batch)


        # Group the previous operations
        group_batch = tf.group(syntax_batch, mutated_syntax_batch)


        # Store datapoints for precision and recall measures
        precision_points = []
        recall_points = []


        # Binary classification decision boundary for precisio recall
        DECISION_THRESHOLD = DECISION_LOWER_BOUND


        # Report testing progress
        class LogProgressHook(tf.train.SessionRunHook):


            # Just before inference
            def before_run(self, run_context):
                return tf.train.SessionRunArgs([syntax_batch, mutated_syntax_batch])


            # Just after inference
            def after_run(self, run_context, run_values):

                # Obtain graph results
                syntax_value, mutated_value = run_values.results


                # Calculations for precision and recall
                false_positives = sum([1. for element in mutated_value if element >= DECISION_THRESHOLD])
                false_negatives = sum([1. for element in syntax_value if element <= DECISION_THRESHOLD])


                # Calculate precision metric true_positive/(true_positive + false_positive)
                precision = BATCH_SIZE / (BATCH_SIZE + false_positives)

                
                # Calculate precision metric true_positive/(true_positive + false_positive)
                recall = BATCH_SIZE / (BATCH_SIZE + false_negatives)


                # Print result for verification
                print(
                    "Threshold: %.2f" % DECISION_THRESHOLD, 
                    "Precision: %.2f" % precision, 
                    "Recall: %.2f" % recall, 
                    "Accurary: %.2f" % ((BATCH_SIZE * 2 - false_positives - false_negatives) / (BATCH_SIZE * 2)))


                # Record current loss
                precision_points.append(precision)
                recall_points.append(recall)


        # Prepare to save and load models
        model_saver = tf.train.Saver()


        # Perform computation cycle based on graph
        with tf.train.MonitoredTrainingSession(hooks=[
            LogProgressHook()]) as session:

            # Load progress from checkpoint
            model_saver.restore(session, model_checkpoint)


            # Repeat testing iteratively for varying decision thresholds
            for i in range(DECISION_RANGE + 1):

                # Run single batch of testing
                _ = session.run(group_batch)

                
                # Calculate new decision threshold
                DECISION_THRESHOLD += (DECISION_UPPER_BOUND/DECISION_RANGE)


     # Construct and save plot
    import matplotlib.pyplot as plt
    plt.scatter(recall_points, precision_points)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(datetime.now().strftime("%Y_%B_%d_%H_%M_%S") + "_syntax_precision_recall.png")
    plt.close()
