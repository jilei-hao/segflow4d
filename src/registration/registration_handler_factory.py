from registration.abstract_registration_handler import AbstractRegistrationHandler


class RegistrationHandlerFactory:
    @staticmethod
    def create_registration_handler(method: str) -> 'AbstractRegistrationHandler':
        if method == 'fireants':
            from registration.fireants.fireants_registration_handler import FireantsRegistrationHandler
            return FireantsRegistrationHandler()
        if method == 'greedy':
            from registration.greedy.greedy_registration_handler import GreedyRegistrationHandler
            return GreedyRegistrationHandler()
        else:
            raise ValueError(f"Unknown registration method: {method}")