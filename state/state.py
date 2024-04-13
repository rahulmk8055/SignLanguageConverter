# class State:
#     def __init__(self, predictedLetter="", currentWord="", confidenceScore=0.0, count=0):
#         self._predictedLetter = predictedLetter
#         self._currentWord = currentWord
#         self._confidenceScore = confidenceScore
#         self._count = count
#
#     # Getters
#     def get_predicted_letter(self):
#         return self._predictedLetter
#
#     def get_current_word(self):
#         return self._currentWord
#
#     def get_confidence_score(self):
#         return self._confidenceScore
#
#     def get_count(self):
#         return self._count
#
#     # Setters
#     def set_predicted_letter(self, predictedLetter):
#         self._predictedLetter = predictedLetter
#
#     def set_current_word(self, currentWord):
#         self._currentWord = currentWord
#
#     def set_confidence_score(self, confidenceScore):
#         if 0.0 <= confidenceScore <= 1.0:
#             self._confidenceScore = confidenceScore
#         else:
#             raise ValueError("Confidence score must be between 0.0 and 1.0")
#
#     def set_count(self, count):
#         if count < 0:
#             raise ValueError("Count cannot be negative")
#         self._count = count
#
#     def append_to_current_word(self, letter):
#         self._currentWord += letter
#     def __str__(self):
#         return (f"Predicted Letter: {self._predictedLetter}\n"
#                 f"Current Word: {self._currentWord}\n"
#                 f"Confidence Score: {self._confidenceScore:.2f}\n"
#                 f"Count: {self._count}")


import json

class State:
    def __init__(self, predictedLetter="", currentWord="", confidenceScore=0.0, count=0):
        self._predictedLetter = predictedLetter
        self._currentWord = currentWord
        self._confidenceScore = confidenceScore
        self._count = count
        self._filename = 'state.json'

    # Getters
    def get_predicted_letter(self):
        return self._predictedLetter

    def get_current_word(self):
        self.load_state()
        return self._currentWord

    def get_confidence_score(self):
        return self._confidenceScore

    def get_count(self):
        return self._count

    # Setters
    def set_predicted_letter(self, predictedLetter):
        self._predictedLetter = predictedLetter
        self.save_state()

    def set_current_word(self, currentWord):
        self._currentWord = currentWord
        self.save_state()

    def set_confidence_score(self, confidenceScore):
        if 0.0 <= confidenceScore <= 1.0:
            self._confidenceScore = confidenceScore
            self.save_state()
        else:
            raise ValueError("Confidence score must be between 0.0 and 1.0")

    def set_count(self, count):
        if count < 0:
            raise ValueError("Count cannot be negative")
        self._count = count
        self.save_state()

    def append_to_current_word(self, letter):
        self._currentWord += letter
        self.save_state()

    def save_state(self):
        # Serialize the state to JSON and save it to a file
        with open(self._filename, 'w') as file:
            json.dump(self.__dict__, file)

    def load_state(self):
        # Load the state from a file and deserialize the JSON
        try:
            with open(self._filename, 'r') as file:
                state_data = json.load(file)
                self._predictedLetter = state_data['_predictedLetter']
                self._currentWord = state_data['_currentWord']
                self._confidenceScore = state_data['_confidenceScore']
                self._count = state_data['_count']
        except FileNotFoundError:
            # If the state file does not exist, initialize with default values
            self.__init__()

    def __str__(self):
        return (f"Predicted Letter: {self._predictedLetter}\n"
                f"Current Word: {self._currentWord}\n"
                f"Confidence Score: {self._confidenceScore:.2f}\n"
                f"Count: {self._count}")
