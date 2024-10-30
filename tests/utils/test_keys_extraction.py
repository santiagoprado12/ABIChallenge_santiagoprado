
from src.utils.keys_extraction import *
from unittest import TestCase
from cryptography.fernet import Fernet
import os


def test_get_secret_key_input(monkeypatch):

    if "SECRET_KEY" in os.environ:
       del os.environ["SECRET_KEY"]

    monkeypatch.setattr('builtins.input', lambda _: "input_string")
    key_extract = KeysExtraction()

    assert key_extract._secret_key == "input_string"


def test_get_secret_key_env_var(monkeypatch):

    monkeypatch.setenv("SECRET_KEY", "env_variable")
    key_extract = KeysExtraction()

    assert key_extract._secret_key == "env_variable"


def test_get_decrypt_file(tmpdir):
    
    def encrypt_and_save_file(fpath, keys):
    
        secret_key = Fernet.generate_key()  
        cipher_suite = Fernet(secret_key)
        encrypted_data = cipher_suite.encrypt(str.encode(json.dumps(keys)))
        with open(fpath, "wb") as file:
            file.write(encrypted_data)
        return secret_key
    
    keys = {
        "access_key": "test_access_key",
        "secret_key": "test_secret_key",
    }

    fpath = f"{tmpdir}/encrypted_data_test.txt"

    secret_key = encrypt_and_save_file(fpath, keys)

    os.environ["SECRET_KEY"] = secret_key.decode()

    key_extract = KeysExtraction(path=fpath)
    decrypted_keys = key_extract.decrypt_file()

    TestCase().assertDictEqual(decrypted_keys, keys)


def test_set_env_variables(monkeypatch):
    
    test_keys = {
        "access_key": "test_access_key",
        "secret_key": "test_secret_key",
    }

    monkeypatch.setattr(KeysExtraction, 'decrypt_file', lambda _: test_keys)
    monkeypatch.setattr(KeysExtraction, 'get_secret_key', lambda _: None)

    key_extract = KeysExtraction()
    key_extract.set_env_variables()

    assert "access_key" in os.environ  #var exists
    assert "secret_key" in os.environ
    assert os.environ["access_key"] == "test_access_key" #var has the correct value
    assert os.environ["secret_key"] == "test_secret_key"



