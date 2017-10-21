# Track number of entries in dataset file
DATASET_LENGTH = 2015538


# Keep track of free cpu cores
CPU_CORES = 8


# Number of operations to chain
RECURSION_DEPTH = 8


# Number of IO examples per dataset row
DATASET_IO_EXAMPLES = 10


# The location on the disk of project
DATASET_BASEDIR = ("C:/Users/brand/Google Drive/" +
    "Academic/Research/Program Synthesis with Deep Learning/Datasets/")


# Path to dataset file
DATASET_FILEPATH = (DATASET_BASEDIR + "dataset.csv")


# Open dataset file write stream
DATASET_FILE = open(DATASET_FILEPATH, "w")


# Deliminator to separate columns
DATASET_DELIMINATOR = ", "


# First column of dataset header
DATASET_HEADER = "function_name"


# Add two header columns for each IO example
for i in range(DATASET_IO_EXAMPLES):
    DATASET_HEADER += (DATASET_DELIMINATOR + "function_input_" + str(i) + 
    DATASET_DELIMINATOR + "function_output_" + str(i))



# Final column of dataset header
DATASET_HEADER += (DATASET_DELIMINATOR + "function_code")


# Initial program to mutate from
DATASET_SEED = "lambda x: x"


# Keep track of current row in dataset
CURSOR_INDEX = 0


# Reset program list for next cycle
def clear_programs():
    DATASET_FILE.truncate()
    DATASET_FILE.write(DATASET_HEADER + "\n")


# Allowed modifications to existing programs
def possible_mutations(current_program):
    return ["+1", "-1", "+x", "-x", "*(x+1)", "/(x+1)"]


# Recursively mutate programs using tree
def mutate_program(current_program, recursive_depth):

    # Generate program for every allowed mutation
    for mutation in possible_mutations(current_program):

        # Mutate existing code
        new_program = current_program + mutation


        # Mutate already mutated code
        if recursive_depth > 0:
            mutate_program(new_program, recursive_depth - 1)


        # Save mutated code as program example
        render_program(new_program)


# Function to render dataset line with multiprocessing
def render_program(current_program):

    # Obntain current cursor position in file
    global CURSOR_INDEX


    # Evaluate program into function F
    F = eval(current_program)


    # Encode column one: name
    current_line = "math_function_" + str(CURSOR_INDEX) + DATASET_DELIMINATOR
    CURSOR_INDEX += 1


    # Encode columns two through twenty-one: IO examples
    for n in range(DATASET_IO_EXAMPLES):
        current_line += str(n) + DATASET_DELIMINATOR + str(F(n)) + DATASET_DELIMINATOR


    # Encode final column: source code
    current_line += current_program


    # Write line to dataset csv file
    DATASET_FILE.write(current_line + "\n")


# Clear existing programs from list
clear_programs()


# Mutate existing program recursively, and generate tree
mutate_program(DATASET_SEED, RECURSION_DEPTH)