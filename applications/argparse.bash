#!/usr/bin/env bash

# Use python's argparse module in shell scripts
#
# The function `argparse` parses its arguments using
# argparse.ArgumentParser; the parser is defined in the function's
# stdin.
#
# Executing ``argparse.bash`` (as opposed to sourcing it) prints a
# script template.
#
# https://github.com/nhoffman/argparse-bash
# MIT License - Copyright (c) 2015 Noah Hoffman


### BEGIN This section added by Steven Potiris

COLOR_RED='\033[0;31m'
COLOR_YELLOW='\033[1;33m'
COLOR_NONE='\033[0m'

function info(){
 echo -e "${COLOR_YELLOW}[$SCRIPTNAME]${COLOR_NONE} ${@}" 2>&1
}

function error(){
 echo -e "${COLOR_RED}[$SCRIPTNAME]${COLOR_NONE} ERROR: ${@}" 2>&1
 exit 1
}

function optional_flag(){
  [[ ! -z $2 ]] && echo "--$1 $2"
}

function bool_flag(){
 local no=no
 [[ ! -z $2 ]]  && no=
 echo "--${no}${1}"
}


function assert (){
  # Usage: assert "condition" "error message"
  # Note: Ensure there are EXACTLY two arguments, and that any paths are enclosed with quotes
  [ $# -ne 2 ] && error "Assert expected exactly two parameters (got $@)."
  local condition="[[ $1 ]]"
  if eval $condition ; then
    [[ $DEBUG ]] && info "Assert passed: $condition" || true
    return 0
  fi
  error "Assertion failed: $condition (${@:2})"
}

#### END

argparse(){
    argparser=$(mktemp 2>/dev/null || mktemp -t argparser)
    cat > "$argparser" <<EOF
from __future__ import print_function
import sys
import argparse
import os


class MyArgumentParser(argparse.ArgumentParser):
    def print_help(self, file=None):
        """Print help and exit with error"""
        super(MyArgumentParser, self).print_help(file=file)
        sys.exit(1)

parser = MyArgumentParser(prog=os.path.basename("$0"),
            description="""$ARGPARSE_DESCRIPTION""", formatter_class=argparse.RawTextHelpFormatter)
EOF

    # stdin to this function should contain the parser definition
    cat >> "$argparser"

    cat >> "$argparser" <<EOF
args = parser.parse_args()
for arg in [a for a in dir(args) if not a.startswith('_')]:
    key = arg.upper()
    value = getattr(args, arg, None)

    if isinstance(value, bool) or value is None:
        print('{0}="{1}";'.format(key, 'yes' if value else ''))
    elif isinstance(value, list):
        print('{0}=({1});'.format(key, ' '.join('"{0}"'.format(s) for s in value)))
    else:
        print('{0}="{1}";'.format(key, value))
EOF

    # Define variables corresponding to the options if the args can be
    # parsed without errors; otherwise, print the text of the error
    # message.
    if python "$argparser" "$@" &> /dev/null; then
        eval $(python "$argparser" "$@")
        retval=0
    else
        python "$argparser" "$@"
        retval=1
    fi

    rm "$argparser"
    return $retval
}

# print a script template when this script is executed
if [[ $0 == *argparse.bash ]]; then
    cat <<FOO
#!/usr/bin/env bash

source \$(dirname \$0)/argparse.bash || exit 1
argparse "\$@" <<EOF || exit 1
parser.add_argument('infile')
parser.add_argument('-o', '--outfile')

EOF

echo "INFILE: \${INFILE}"
echo "OUTFILE: \${OUTFILE}"
FOO
fi
