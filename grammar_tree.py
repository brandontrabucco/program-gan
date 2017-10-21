# Possible mutations to initial function
DATASET_MUTATIONS = ["+1", "-1", "+x", "-x", "*(x+1)", "/(x+1)"]
print("Allowed mutations:", DATASET_MUTATIONS)


# Number of operations to chain
RECURSION_DEPTH = 7
print("Grammar tree depth:", RECURSION_DEPTH)


# Track number of entries in dataset file
DATASET_LENGTH = sum([len(DATASET_MUTATIONS)**(i + 1) for i in range(RECURSION_DEPTH)])
print("Total number rows:", DATASET_LENGTH)


# Initial program to mutate from
DATASET_SEED = "lambda x: x"
print("Initial function seed:", DATASET_SEED)


# Number of IO examples per dataset row
DATASET_IO_EXAMPLES = 10


# Path to dataset file
DATASET_FILEPATH = ("epf_" + str(RECURSION_DEPTH) + "_dataset.csv")


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


# Keep track of current row in dataset
CURSOR_INDEX = 0


# Reset program list for next cycle
def clear_programs():
    DATASET_FILE.truncate()
    DATASET_FILE.write(DATASET_HEADER + "\n")


# Allowed modifications to existing programs
def possible_mutations(current_program):
    return DATASET_MUTATIONS


# Recursively mutate programs using tree
def mutate_program(current_program, recursive_depth):
    
    # Check if enough mutations have been made
    if recursive_depth > 0:

        # Generate program for every allowed mutation
        for mutation in possible_mutations(current_program):

            # Mutate existing code
            new_program = current_program + mutation


            # Mutate already mutated code
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
print("Cleaning dataset file.")
clear_programs()


# Mutate existing program recursively, and generate tree
print("Generating grammar tree.")
mutate_program(DATASET_SEED, RECURSION_DEPTH)
print("Finished.")