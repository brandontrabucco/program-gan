import multiprocessing as mp
import threading as td

	
# Force function to block single threads
def synchronized(function_to_block):
    function_to_block.__lock__ = td.Lock()
    def blocking_function(*args, **kws):
        with function_to_block.__lock__:
            return function_to_block(*args, **kws)
    return blocking_function


# Keep track of free cpu cores
CPU_CORES = 8


# Number of operations to chain
RECURSION_DEPTH = 8


# The location on the disk of project
DATASET_BASEDIR = ("C:/Users/brand/Google Drive/" +
    "Academic/Research/Program Synthesis with Deep Learning/Datasets/")


# Path to dataset file
DATASET_FILEPATH = DATASET_BASEDIR + "dataset.csv"


# Open dataset file write stream
DATASET_FILE = open(DATASET_FILEPATH, "w")


# Deliminator to separate columns
DATASET_DELIMINATOR = ", "


# Header to dataset csv file
DATASET_HEADER = ("function_name" + 
    DATASET_DELIMINATOR + "function_input_0" + 
    DATASET_DELIMINATOR + "function_output_0" + 
    DATASET_DELIMINATOR + "function_input_1" + 
    DATASET_DELIMINATOR + "function_output_1" + 
    DATASET_DELIMINATOR + "function_input_2" + 
    DATASET_DELIMINATOR + "function_output_2" + 
    DATASET_DELIMINATOR + "function_input_3" + 
    DATASET_DELIMINATOR + "function_output_3" + 
    DATASET_DELIMINATOR + "function_input_4" + 
    DATASET_DELIMINATOR + "function_output_4" + 
    DATASET_DELIMINATOR + "function_input_5" + 
    DATASET_DELIMINATOR + "function_output_5" + 
    DATASET_DELIMINATOR + "function_input_6" + 
    DATASET_DELIMINATOR + "function_output_6" + 
    DATASET_DELIMINATOR + "function_input_7" + 
    DATASET_DELIMINATOR + "function_output_7" + 
    DATASET_DELIMINATOR + "function_input_8" + 
    DATASET_DELIMINATOR + "function_output_8" + 
    DATASET_DELIMINATOR + "function_input_9" + 
    DATASET_DELIMINATOR + "function_output_9" + 
    DATASET_DELIMINATOR + "function_code")


# Convert string programs to executable function
DATASET_ENTRIES = ["lambda x: x"]


# Keep track of current row in dataset
CURSOR_INDEX = 0


# Reset program list for next cycle
def clear_programs():
    DATASET_ENTRIES.clear()
    DATASET_ENTRIES.append("lambda x: x")
    DATASET_FILE.truncate()
    insert_row_dataset(DATASET_HEADER)
    CURSOR_INDEX = 0


# Track current row index in dataset file
@synchronized
def increment_cursor():
    global CURSOR_INDEX
    CURSOR_INDEX += 1
    return (CURSOR_INDEX - 1)


# Write single line to dataset csv file
@synchronized
def insert_row_dataset(row_to_insert):
    DATASET_FILE.write(row_to_insert + "\n")


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


        # Save mutated code an program example
        DATASET_ENTRIES.append(new_program)


# Function to render dataset line with multiprocessing
def render_program(current_program):

    # Evaluate program into function F
    F = eval(current_program)


    # Encode column one: name
    current_line = "math_function_" + str(increment_cursor()) + DATASET_DELIMINATOR


    # Encode columns two through twenty-one: IO examples
    for n in range(10):
        current_line += str(n) + DATASET_DELIMINATOR + str(F(n)) + DATASET_DELIMINATOR


    # Encode final column: source code
    current_line += current_program


    # Write line to dataset csv file
    insert_row_dataset(current_line)


# Clear existing programs from list
clear_programs()


# Mutate existing program recursively, and generate tree
mutate_program(DATASET_ENTRIES[0], RECURSION_DEPTH)


if __name__ == "__main__":

    # Create a handle with multiple cores
    multi_core = mp.Pool(CPU_CORES)

    # Execute render function on every program
    multi_core.map(render_program, DATASET_ENTRIES)