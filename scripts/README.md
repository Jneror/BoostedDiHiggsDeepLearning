# Bash scripts

## Generate root datasets

Merge signal roots with background ones.

### Requirements

* Root environment activated
* hadd shell command available

### Execute

In the folder containing the script run:

```console
foo@bar:~$ ./gen_root_datasets.sh <signal> <root_files_dir> <out_dir>
```

For example, for our project we run:

```console
foo@bar:~$ ./gen_root_datasets.sh Xtohh ../raw_data ../processed_data
```