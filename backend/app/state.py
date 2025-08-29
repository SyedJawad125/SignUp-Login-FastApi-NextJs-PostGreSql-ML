# app/state.py

class TrainingState:
    def __init__(self):
        self.status = "idle"        # idle, training, completed, failed
        self.message = None

    def set_status(self, status: str, message: str = None):
        self.status = status
        self.message = message

# ✅ single global instance
training_state = TrainingState()
