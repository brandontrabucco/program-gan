import multiprocessing as mp


# Keep track of free cpu cores
CPU_CORES = 8


# Number of operations to chain
RECURSION_DEPTH = 8


# Open dataset file write stream
DATASET_FILE = open("dataset.csv", "w")


# Convert string programs to executable function
PROGRAM_LIST = ["lambda x: x"]


# Reset program list for next cycle
def clear_programs():
    PROGRAM_LIST.clear()
    PROGRAM_LIST.append("lambda x: x")
    DATASET_FILE.truncate()


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
        PROGRAM_LIST.append(new_program)


# Function to render dataet line with multiprocessing
def render_program(current_program):

    # Evaluate program into function F
    F = eval(current_program)


    # Encode column one: name
    current_line = "grammer_tree_program" + ", "


    # Encode columns two through twenty-one: IO examples
    for n in range(10):
        current_line += str(n) + ", " + str(F(n)) + ", "


    # Encode final column: source code
    current_line += current_program + "\n"


    # Write line to dataset csv file
    DATASET_FILE.write(current_line)


# Clear existing programs from list
clear_programs()


# Mutate existing program recursively, and generate tree
mutate_program(PROGRAM_LIST[0], RECURSION_DEPTH)


if __name__ == "__main__":

    # Create a handle with multiple cores
    multi_core = mp.Pool(CPU_CORES)

    # Execute render function on every program
    multi_core.map(render_program, PROGRAM_LIST)