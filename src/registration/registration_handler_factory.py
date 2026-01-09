from registration.abstract_registration_handler import AbstractRegistrationHandler


class RegistrationHandlerFactory:

    @staticmethod
    def create_registration_handler(backend) -> 'AbstractRegistrationHandler':
        """Create a new registration handler based on the backend type"""
        if backend == 'fireants':
            from registration.fireants.fireants_registration_handler import FireantsRegistrationHandler
            return FireantsRegistrationHandler()
        elif backend == 'greedy':
            from registration.greedy.greedy_registration_handler import GreedyRegistrationHandler
            return GreedyRegistrationHandler()
        else:
            raise ValueError(f"Unknown registration backend: {backend}")