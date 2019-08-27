import os
import shutil
import subprocess
import hashlib


def hash_kwargs(kwargs):
    kwargs = tuple(sorted(list(kwargs.items())))
    kwargs_hash = hashlib.sha224(repr(kwargs)).hexdigest()
    return kwargs_hash


class Sentinal(object):
    def __init__(self, sentinal_filename):
        self.sentinal_filename = sentinal_filename
    @property
    def unfinished(self):
        if os.path.exists(self.sentinal_filename):
            print ('sentinal file ' + self.sentinal_filename + ' exists')
            return False
        return True
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            with open(self.sentinal_filename, 'w') as sentinal_file:
                pass


class SentinalFactory(object):
    def __init__(self, filename_prefix, kwargs):
        self.filename_prefix = filename_prefix
        self.kwargs_hash = hash_kwargs(kwargs)
    def __call__(self, name):
        return Sentinal(self.filename_prefix + name + '_' + self.kwargs_hash)


def makedirs(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != 17:
            raise


def rmtree(directory):
    try:
        shutil.rmtree(directory)
    except OSError as e:
        if e.errno != 2:
            raise


def remove(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != 2:
            raise


def symlink(filename, link_name=None, link_directory=None):
    if link_name is None:
        link_name = os.path.basename(filename)
    if link_directory is None:
        link_directory = os.getcwd()
    link_filename = os.path.join(link_directory, link_name)
    remove(link_filename)
    filename = os.path.abspath(filename)
    os.symlink(filename, link_filename)
    return link_filename


class CurrentDirectory(object):
    def __init__(self, directory):
        self.directory = directory
    def __enter__(self):
        self.prev_directory = os.getcwd()
        makedirs(self.directory)
        os.chdir(self.directory)
    def __exit__(self, *args):
        os.chdir(self.prev_directory)


def wget_file(url, filename):
    makedirs(os.path.dirname(filename))
    subprocess.check_call(['wget', '--no-check-certificate', url, '-O', filename])


def wget_file_gunzip(url, filename):
    makedirs(os.path.dirname(filename))
    subprocess.check_call(['wget', '--no-check-certificate', url, '-O', filename+'.gz'])
    remove(filename)
    subprocess.check_call(['gunzip', filename+'.gz'])


class SafeWriteFile(object):
    def __init__(self, filename):
        self.filename = filename
        self.temp_filename = filename + '.tmp'
    def __enter__(self):
        return self.temp_filename
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            os.rename(self.temp_filename, self.filename)


class InvalidInitParam(Exception):
    pass



