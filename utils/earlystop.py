class EarlyStopper:
    def __init__(self, patience=10, min_delta=0) -> None:
        """Initializes EarlyStopper class.

        Args:
            patience (int, optional): The amount of epochs you want to wait to early stop. Defaults to 1.
            min_delta (int, optional): Minimum difference to tolerate. Defaults to 0.
        """

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss: float) -> bool:
        """Determines whether to early stop.

        Args:
            validation_loss (float): The validation loss.

        Returns:
            bool: Whether to early stop.
        """

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
