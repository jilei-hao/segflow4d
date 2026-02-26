from segflow4d.registration.registration_handler.abstract_registration_handler import AbstractRegistrationHandler


class RegistrationHandlerFactory:

    @staticmethod
    def create_registration_handler(backend) -> 'AbstractRegistrationHandler':
        """Create a new registration handler based on the backend type"""
        match backend:
            case 'fireants':
                from segflow4d.registration.registration_handler.fireants.fireants_registration_handler import FireantsRegistrationHandler
                return FireantsRegistrationHandler()
            case 'greedy':
                from segflow4d.registration.registration_handler.greedy.greedy_registration_handler import GreedyRegistrationHandler
                return GreedyRegistrationHandler()
            case _:
                raise ValueError(f"Unknown registration backend: {backend}")