#!/usr/bin/env bash
set -uo pipefail

ALLOWED="src/eval/test_gate.py tests/test_test_gate.py"

# Quét trong src tests
HITS=$(grep -RIl "P_test\|Y_test\|_test\.npy" src tests --include=*.py 2>/dev/null | sort)
STATUS=0

for f in $HITS; do
  case " $ALLOWED " in
    *" $f "*) ;;
    *) echo "VI PHAM cach ly Test: $f"; STATUS=1 ;;
  esac
done

if grep -RIn "make_objective" src/optimizers --include=*.py 2>/dev/null; then
  echo "VI PHAM: make_objective xuat hien trong src/optimizers"; STATUS=1
fi

if grep -RInE "np\.random\.seed|[^_a-zA-Z.]hash\(" src --include=*.py 2>/dev/null; then
  echo "VI PHAM: dung np.random.seed hoac hash()"; STATUS=1
fi

[ $STATUS -eq 0 ] && echo "Cach ly Test: OK"
exit $STATUS