from dataclasses import dataclass


@dataclass
class ActHistogram:
    """
    A dataclass for storing arbitrary activation data via hooks.
    The data is stored in `data`, and `hook_handles` stores the hook handles.
    """

    data: dict
    hook_handles: dict

    def reset(self):

        # Remove activation hook once we're done with them
        # Otherwise the hook will remain in, slowing down forward calls
        for handle in self.hook_handles.values():
            handle.remove()

        self.data = {}
        self.hook_handles = {}
