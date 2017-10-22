import tensorflow as tf
import string as sn


# The location on the disk of project
PROJECT_BASEDIR = ("C:/Users/brand/Google Drive/" +
    "Academic/Research/Program Synthesis with Deep Learning/Repo/program-gan/")


# The location on the disk of checkpoints
CHECKPOINT_BASEDIR = (PROJECT_BASEDIR + "Checkpoints/")


# The location on the disk of project
DATASET_BASEDIR = ("C:/Users/brand/Google Drive/" +
    "Academic/Research/Program Synthesis with Deep Learning/Datasets/")


# Filenames associated with program dataset
DATASET_FILENAMES_PYTHON = [
    (DATASET_BASEDIR + "epf_8_dataset.csv")
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


    # Convert integer lookup table
    lookup_table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_characters.values, default_value=0)


    # Query lookup table
    one_hot_tensor = tf.one_hot(lookup_table.lookup(input_characters.values), len(vocabulary), dtype=tf.float32)


    # Calculate actual sequence length
    actual_length = tf.size(one_hot_tensor) // len(vocabulary)


    # Pad input to match DATASET_MAXIMUM
    expanded_tensor = tf.pad(one_hot_tensor, [[0, (DATASET_MAXIMUM - actual_length)], [0, 0]])

    return tf.reshape(expanded_tensor, [DATASET_MAXIMUM, len(vocabulary)]), actual_length


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
BATCH_SIZE = 32
NUM_THREADS = 4
TOTAL_EXAMPLES = 2015538
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


# Compute syntax label with rnn
def inference_syntax_python(program_batch):

    return tf.constant([1. for _ in range(BATCH_SIZE)], dtype=tf.float32)


# Compute loss for syntax discriminator
def loss(prediction, labels):

    # Calculate huber loss of prediction
    huber_loss = tf.losses.huber_loss(labels, prediction)


    # Calculate the mean loss across batch
    huber_loss_mean = tf.reduce_mean(huber_loss)
    tf.add_to_collection(COLLECTION_LOSSES, huber_loss_mean)

    return huber_loss_mean


# Hyperparameters
INITIAL_LEARNING_RATE = 0.001
DECAY_STEPS = 10 * EPOCH_SIZE
DECAY_FACTOR = 0.95


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
def train_epf_8(num_epoch=1):

    # Watch compute time per batch
    from time import time
    from datetime import datetime


    # Convert epoch to batch steps
    num_steps = num_epoch * EPOCH_SIZE


    # Create new graph
    with tf.Graph().as_default():

        # Compute single training batch
        name_batch, examples_batch, program_batch, length_batch = training_batch_python()


        # Compute corrected code
        corrected_batch = inference_generator_python(program_batch)


        # Compute syntax of corrected code
        syntax_batch = inference_syntax_python(corrected_batch)
        syntax_loss = loss(syntax_batch, tf.constant([1. for _ in range(BATCH_SIZE)], tf.float32))


        # Compute behavior of corrected code
        behavior_batch = inference_behavior_python(corrected_batch)
        behavior_loss = []
        for n in range(DATASET_IO_EXAMPLES):
            input_example = tf.constant([n for _ in range(BATCH_SIZE)], tf.float32)
            output_prediction = behavior_batch(input_example)
            behavior_loss.append(loss(output_prediction, input_example))
        behavior_loss = tf.stack(behavior_loss)


        # Obtain total loss of entire model
        total_loss = tf.add_n(tf.get_collection(COLLECTION_LOSSES), name=(PREFIX_TOTAL + EXTENSION_LOSS))
        gradient_batch = train(total_loss)


        # Store datapoints each epoch for plot
        data_points = []


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
                return tf.train.SessionRunArgs([syntax_batch, syntax_loss])


            # Just after inference
            def after_run(self, run_context, run_values):

                # Calculate weighted speed
                self.batch_speed = (0.2 * self.batch_speed) + (0.8 * (1.0 / (time() - self.start_time + 1e-7)))


                # Update every period of steps
                if (self.current_step % EPOCH_SIZE == 0):

                    # Obtain graph results
                    syntax_value, loss_value = run_values.results


                    # Perform some computation, e.g. export a plot, or calculate accuracy
                    average_score = tf.reduce_mean(syntax_value)


                    # Display date, batch speed, estimated time, loss, and accuracy
                    print(
                        datetime.now(),
                        "CUR: %d" % self.current_step,
                        "REM: %d" % (num_steps - self.current_step),
                        "SPD: %.2f bat/sec" % self.batch_speed,
                        "ETA: %.2f hrs" % ((num_steps - self.current_step) / self.batch_speed / 60 / 60),
                        "L: %.2f" % loss_value)


                    # Record current loss
                    data_points.append(loss_value)


        # Prepare to save and load models
        model_saver = tf.train.Saver()


        # Perform computation cycle based on graph
        with tf.train.MonitoredTrainingSession(hooks=[
            tf.train.StopAtStepHook(num_steps=num_steps),
            tf.train.CheckpointSaverHook(CHECKPOINT_BASEDIR,
                save_steps=EPOCH_SIZE,
                saver=model_saver),
            LogProgressHook()]) as session:

            # Repeat training iteratively
            while not session.should_stop():

                # Run single batch of training
                session.run(gradient_batch)


     # Construct and save plot
    import matplotlib.pyplot as plt
    plt.plot(data_points)
    plt.xlabel("Training Epoch")
    plt.ylabel("Mean Huber Syntax Loss")
    plt.savefig(datetime.now().strftime("%Y_%B_%d_%H_%M_%S") + "_syntax_training_loss.png")
    plt.close()
