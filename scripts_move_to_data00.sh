#!/bin/bash
# Move bulky derived/superseded data to /stage/irsa-jointproc-data00/JAISP, verify, symlink back.
set -e
SRC=/home/shemmati/Work/Projects/JAISP/data
DST=/stage/irsa-jointproc-data00/JAISP
mkdir -p $DST
LOG=/home/shemmati/Work/Projects/JAISP/io/_nb23_outputs/clampfix_harness/move_data00.log

move_one () {
  NAME=$1
  echo "=== $NAME: rsync ===" >> $LOG
  rsync -a $SRC/$NAME/ $DST/$NAME/ >> $LOG 2>&1
  echo "=== $NAME: verify (2nd pass must transfer nothing) ===" >> $LOG
  NDIFF=$(rsync -a -n --out-format='%n' $SRC/$NAME/ $DST/$NAME/ | wc -l)
  NSRC=$(find $SRC/$NAME -type f | wc -l); NDST=$(find $DST/$NAME -type f | wc -l)
  echo "$NAME: second-pass diffs=$NDIFF files src=$NSRC dst=$NDST" >> $LOG
  if [ "$NDIFF" -ne 0 ] || [ "$NSRC" -ne "$NDST" ]; then
    echo "$NAME: VERIFY FAILED - originals untouched" >> $LOG; return 1
  fi
  mv $SRC/$NAME $SRC/${NAME}_PENDING_DELETE
  ln -s $DST/$NAME $SRC/$NAME
  # prove the symlink serves a real file before deleting
  F=$(find -L $SRC/$NAME -type f | head -1)
  if [ -n "$F" ] && [ -s "$F" ]; then
    rm -rf $SRC/${NAME}_PENDING_DELETE
    echo "$NAME: MOVED, symlinked, originals deleted" >> $LOG
  else
    echo "$NAME: symlink check FAILED - originals kept at ${NAME}_PENDING_DELETE" >> $LOG; return 1
  fi
}

move_one euclid_tiles_all
move_one cached_features_v10_q1
echo "ALL MOVES DONE" >> $LOG
df -h /home/shemmati >> $LOG
