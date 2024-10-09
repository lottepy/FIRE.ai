import sys
from os import remove
import shutil
from cryptography.fernet import Fernet, InvalidToken
from hashlib import sha256

def encrypt(filename, overwrite=True, manual=False, filekey=None):
    print(f"---- Encryption of {filename} ----")

    generated_filekey_name = 'filekey.txt'

    # Filekey generation
    if manual == False:
        print("Filekey generation...")
        key = Fernet.generate_key()
        with open(generated_filekey_name, 'wb') as filekey:
            filekey.write(key)

    # Manual mode = no filekey generation
    else:
        generated_filekey_name = filekey

    # Filekey reading and retrieved as bytes
    print("Filekey reading...")
    try:
        with open(generated_filekey_name, 'rb') as filekey:
            key = filekey.read()  # key = bytes
    except FileNotFoundError:
        print(f"Error : No such keyfile : {filename} in the current folder")
        return None

    # Fernet object creation with key
    f = Fernet(key)

    # Copying the file before overwriting it
    if overwrite == False:
        name = filename[:filename.find('.')]
        name += "(copy)"
        ext = filename[filename.find('.'):]
        copy_filename = name + ext
        print("File copy before overwriting...")
        try:
            shutil.copyfile(filename, copy_filename)
        except FileNotFoundError:
            print(f"Error : No such file : {filename} in the current folder")
            remove(generated_filekey_name)
            return None

    # Recovery the file to be encrypted in bytes
    print(f"{filename} reading...")
    try:
        with open(filename, 'rb') as file:
            file_bytes = file.read()  # file_bytes = bytes
    except FileNotFoundError:
        print(f"Error : No such file : {filename} in the current folder")
        remove(generated_filekey_name)
        return None

    # Bytes data encryption
    print(f"Data encryption...")
    encrypted = f.encrypt(file_bytes)

    # Overwriting the file with encrypted bytes data
    print(f"Data writing...")
    with open(filename, 'wb') as encrypted_file:
        encrypted_file.write(encrypted)

    print("---- Operation completed successfully ----")

    if manual == False:
        print("A keyfile.key file has been generated in the current folder, please keep it safe")


def decrypt(filename, filekey_name):
    print(f"---- Decryption of {filename} ----")

    # Filekey reading and retrieved as bytes
    print("Filekey reading...")
    try:
        with open(filekey_name, 'rb') as filekey:
            key = filekey.read()  # key = bytes
    except FileNotFoundError:
        print(f"Error : No such filekey : '{filekey_name}' in the current folder")
        return None

    # Fernet object creation with key
    f = Fernet(key)

    # Recovery the file to be decrypted in bytes
    print(f"{filename} reading...")
    try:
        with open(filename, 'rb') as encrypted_file:
            encrypted = encrypted_file.read()  # encrypted = bytes
    except FileNotFoundError:
        print(f"Error : No such file : '{filename}' in the current folder")
        return None

    # Bytes data decryption
    print("Decrypting data...")
    try:
        decrypted = f.decrypt(encrypted)
    except InvalidToken:
        print("Error : Invalid keyfile")
        return None

    # Overwriting the file with decrypted bytes data
    # File regains its integrity
    print(f"{filename} writing...")
    with open(filename, 'wb') as decrypted_file:
        decrypted_file.write(decrypted)

    print("Operation completed successfully")


if __name__ == "__main__":
    try:
        # ENCRYPT
        # 1 - encrypt
        # 2 - filename
        # 3 - ow - c
        if sys.argv[1] == 'encrypt':
            if sys.argv[3] == 'ow':  # Overwriting
                encrypt(sys.argv[2], overwrite=True)
            elif sys.argv[3] == 'c':  # Copy before overwriting
                encrypt(sys.argv[2], overwrite=False)
            else:  # ERROR
                print("Error : last argument of encrypt function must be 'ow' or 'c'")

        # ENCRYPT_MANUAL
        # 1 - encryptm
        # 2 - filename
        # 3 - filekey
        # 4 - ow - c
        elif sys.argv[1] == 'encryptm':
            if sys.argv[4] == 'ow':  # Overwriting
                encrypt(sys.argv[2], overwrite=True,
                        manual=True, filekey=sys.argv[3])

            elif sys.argv[4] == 'c':  # Copy before overwriting
                encrypt(sys.argv[2], overwrite=False,
                        manual=True, filekey=sys.argv[3])

            else:  # ERROR
                print("Error : last argument of encrypt function must be 'ow' or 'c'")

                # DECRYPT
        # 1 - decrypt
        # 2 - filename
        # 3 - filekey
        elif sys.argv[1] == 'decrypt':
            decrypt(sys.argv[2], sys.argv[3])

        # ERROR
        else:
            print(f"Error : The 1st argument must be 'encrypt', 'decrypt' or 'encryptm'. Given : '{sys.argv[1]}'")

    except IndexError:  # Wrong parameter order
        print("Error : parameters order must be :")
        print("- encrypt filename ow/c")
        print("- encryptm filename keyfile_name ow/c")
        print("- decrypt filename keyfile_name")