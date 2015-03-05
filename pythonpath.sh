#
# Adds the current directory to the python path
#

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH=$DIR:$PYTHONPATH
