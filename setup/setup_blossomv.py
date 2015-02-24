import os
import subprocess

if not os.path.exists('blossom5-v2.04.src.tar.gz'):

    subprocess.check_call(['wget', 'http://pub.ist.ac.at/~vnk/software/blossom5-v2.04.src.tar.gz'])

if not os.path.exists('blossom5-v2.04.src'):

    subprocess.check_call(['tar', '-xzvf', 'blossom5-v2.04.src.tar.gz'])

os.chdir('blossom5-v2.04.src')

subprocess.check_call(['make'])

