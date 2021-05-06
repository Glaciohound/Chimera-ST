import sys
import pdb


def pdb_on_exception(exception_class, exception, traceback):
    pdb.post_mortem(traceback)


if '--pdb-on-error' in sys.argv:
    sys.excepthook = pdb_on_exception
    sys.argv.remove('--pdb-on-error')
elif '--pdb' in sys.argv:
    sys.argv.remove('--pdb')
    pdb.set_trace()
