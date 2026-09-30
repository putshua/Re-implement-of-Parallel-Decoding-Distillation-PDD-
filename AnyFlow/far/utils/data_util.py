import random


# retry random fetch data when failed to read
def retry_load_error(max_attempts=3):
    def decorator(func):
        def wrapper(self, index):
            last_err = None
            for attempt in range(max_attempts):
                try:
                    return func(self, index)
                except Exception as e:
                    last_err = e
                    index = random.randint(0, len(self) - 1)

            raise RuntimeError(f'Dataset failed after {max_attempts} attempts. Last error: {last_err}') from last_err
        return wrapper
    return decorator
