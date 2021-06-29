from enums.perturbator_enum import PerturbatorEnum

yes_answers = ["Yes", "yes", "Y", "y"]
no_answers = ["No", "no", "N", "n"]


def get_yes_no_input(text: str, yes_default: bool):
    while True:
        print(text, end=' ')
        if yes_default:
            inp = input("(Yes/no) >> ")
        else:
            inp = input("(yes/No) >> ")

        if inp == '':
            if yes_default:
                return True
            else:
                return False
        elif inp in yes_answers:
            return True
        elif inp in no_answers:
            return False
        else:
            continue


def get_int_input(text: str, range_start_val: int, range_end_val: int):
    while True:
        try:
            if text is not '':
                print(text)
            inp = int(input("enter number in range (" + str(range_start_val) + "~" + str(range_end_val) + ") >> "))
        except Exception as e:
            print(e)
            continue
        if inp in range(range_start_val, range_end_val + 1):
            return inp
        else:
            print("An out-of-range value was entered")
            continue


def get_float_input(text: str, range_start_val: float, range_end_val: float):
    while True:
        try:
            if text is not '':
                print(text)
            inp = float(input("enter number in range (" + str(range_start_val) + "~" + str(range_end_val) + ") >> "))
        except Exception as e:
            print(e)
            continue
        if range_start_val <= inp <= range_end_val:
            return inp
        else:
            print("An out-of-range value was entered")
            continue


def get_perturbation_percentage():
    perturbator_percentage = []

    print("\nSet perturbator percentage")
    print("0. no perturbation")
    print("1. blackout only")
    print("2. whiteout only")
    print("3. smoothing only")
    print("4. binarization only")
    print("5. edge detection only")
    print("6. hybrid")
    print("7. customize percentage")

    preset = get_int_input('', 0, 7)
    preset = PerturbatorEnum(preset)

    if preset == PerturbatorEnum.NO_PERTURBATION:
        perturbator_percentage = [0, 0, 0, 0, 0]
    elif preset == PerturbatorEnum.BLACKOUT:
        perturbator_percentage = [1, 0, 0, 0, 0]
    elif preset == PerturbatorEnum.WHITEOUT:
        perturbator_percentage = [0, 1, 0, 0, 0]
    elif preset == PerturbatorEnum.SMOOTHING:
        perturbator_percentage = [0, 0, 1, 0, 0]
    elif preset == PerturbatorEnum.BINARIZATION:
        perturbator_percentage = [0, 0, 0, 1, 0]
    elif preset == PerturbatorEnum.EDGE_DETECTION:
        perturbator_percentage = [0, 0, 0, 0, 1]
    elif preset == PerturbatorEnum.HYBRID:
        perturbator_percentage = [0.2, 0.2, 0.2, 0.2, 0.2]
    else:
        while True:
            print("Sum of percentage must be 1")
            perturbator_percentage.append(get_float_input("blackout percentage", 0, 1))
            perturbator_percentage.append(get_float_input("whiteout percentage", 0, 1))
            perturbator_percentage.append(get_float_input("smoothing percentage", 0, 1))
            perturbator_percentage.append(get_float_input("binarization percentage", 0, 1))
            perturbator_percentage.append(get_float_input("edge detection percentage", 0, 1))

            if sum(perturbator_percentage) == 1:
                break

    return perturbator_percentage
