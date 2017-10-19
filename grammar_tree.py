# Convert string programs to executable function
PROGRAM_FORMAT = (lambda program: "(lambda x: {0})".format(program))
PROGRAM_LIST = ["x"]


# Reset program list for next cycle
def clear_programs():
    PROGRAM_LIST.clear()
    PROGRAM_LIST.append("x")


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


# COstruct a tree of given depth using mutate_program, and render dataset csv
def construct_tree(recursive_depth):

    # Clear existing programs from list
    clear_programs()

    # Query the initial program, identity function
    original_program = PROGRAM_LIST[0]

    # Mutate existing program recursively, and generate tree
    mutate_program(original_program, recursive_depth)

    # Render programs to dataset csv
    with open("dataset.csv", "w") as output_file:

        # Render every program in list
        for i in range(len(PROGRAM_LIST)):

            # Translate program to executable format
            current_program = PROGRAM_FORMAT(PROGRAM_LIST[i])

            # Evaluate program into function F
            F = eval(current_program)

            # Encode column one: name
            current_line = "grammer_tree_program_" + str(i) + ", "

            # Encode columns two through twenty-one: IO examples
            for n in range(10):
                current_line += str(n) + ", " + str(F(n)) + ", "

            # Encode final column: source code
            current_line += current_program + "\n"

            # Write line to dataset csv file
            output_file.write(current_line)
