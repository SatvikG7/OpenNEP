import colorama
from colorama import Fore, Back, Style

from helpers.config import get_models


def init() -> str:
    """
    Function to initialize the chatbot and prompt the user for the model to use.

    Returns:
        model_name: str
    """
    colorama.init()
    print(
        Fore.CYAN
        + "Welcome to the OPENNEP chatbot! Please enter the following details:"
        + Style.RESET_ALL
    )
    print(Fore.YELLOW + "Choose from the following models: " + Style.RESET_ALL)

    models_dict = get_models()

    print(Fore.GREEN + str(list(models_dict.keys())) + Style.RESET_ALL)


    model_name = input(Fore.LIGHTWHITE_EX + "Model: " + Style.RESET_ALL)

    while model_name not in list(models_dict.keys()):
        print(
            Fore.RED
            + "Invalid model name. Please choose from the following models: "
            + Style.RESET_ALL
        )
        print(Fore.GREEN + str(list(models_dict.keys())) + Style.RESET_ALL)
        model_name = input(Fore.LIGHTWHITE_EX + "Model: " + Style.RESET_ALL)

    print(Fore.GREEN + "Model selected: " + model_name + Style.RESET_ALL)
    print(Fore.CYAN + "Initializing chatbot..." + Style.RESET_ALL)

    return model_name
