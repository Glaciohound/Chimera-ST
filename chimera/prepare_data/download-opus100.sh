while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data-dir) export DATA_ROOT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
echo "arguments"
echo "DATA_ROOT: $DATA_ROOT"
echo

mkdir -p $DATA_ROOT/orig
cd $DATA_ROOT/orig
tarfile=opus-100-corpus-v1.0.tar.gz

wget -P $DATA_ROOT https://object.pouta.csc.fi/OPUS-100/v1.0/$tarfile
tar xzvf $tarfile
