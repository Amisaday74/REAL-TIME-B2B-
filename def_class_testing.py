# Define a class to represent a Book
class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author
        self.is_checked_out = False  # Every book starts as available

    def checkout(self):
        if not self.is_checked_out:
            self.is_checked_out = True
            return f"You have checked out '{self.title}'."
        else:
            return f"Sorry, '{self.title}' is already checked out."

    def return_book(self):
        if self.is_checked_out:
            self.is_checked_out = False
            return f"'{self.title}' has been returned. Thank you!"
        else:
            return f"'{self.title}' was not checked out."


# Define a function outside the class to display book information
def display_book_info(book):
    status = "Checked Out" if book.is_checked_out else "Available"
    print(f"Title: {book.title}, Author: {book.author}, Status: {status}")


# Example usage
my_book = Book("The Great Gatsby", "F. Scott Fitzgerald")

display_book_info(my_book)          # Show info
print(my_book.checkout())           # Check out the book
display_book_info(my_book)          # Show updated info
print(my_book.return_book())        # Return the book
display_book_info(my_book)          # Show updated info again
