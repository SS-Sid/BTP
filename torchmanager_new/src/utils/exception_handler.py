"""module to handle exceptions throughout the DL project structure.
"""

import sys


def error_message_detail(
        error, 
        error_detail : sys
):
    _, _, exc_tb = error_detail.exc_info()

    error_message = f"Error occurred in {exc_tb.tb_frame.f_code.co_filename} \
          at line {exc_tb.tb_lineno} with error message: {str(error)}"



class CustomException(Exception):
    """Custom exception class to handle exceptions throughout the DL project structure.
    """
    def __init__(
            self, 
            error, 
            error_detail : sys
    ):
        super().__init__(error_message_detail(error, error_detail))
        self.error = error
        self.error_detail = error_detail


    def __str__(self):
        return self.error_message_detail(self.error, self.error_detail)