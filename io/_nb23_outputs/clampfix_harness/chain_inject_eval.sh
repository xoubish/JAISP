#!/bin/bash
# Waits for the overnight driver to finish (ALL DONE = vis_peak v11 trained), then runs
# the star-donor injection eval: v11 twins on GPU0, v10 base vis_peak on GPU0 after.
cd /home/shemmati/Work/Projects/JAISP
L=logs/q1_detection_v11
until grep -q "ALL DONE" logs/q1_detection_v11_driver.log 2>/dev/null; do sleep 300; done
echo "[chain] driver done -> injection eval (v11 twins)" >> $L/chain.log
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=models python3 io/_nb23_outputs/clampfix_harness/eval_inject_v11.py v11 \
  >> $L/inject_eval_v11.log 2>&1
echo "[chain] v11 twins eval done -> v10 base vis_peak eval" >> $L/chain.log
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=models python3 io/_nb23_outputs/clampfix_harness/eval_inject_v11.py v10base \
  >> $L/inject_eval_v10base.log 2>&1
echo "[chain] ALL INJECTION EVALS DONE" >> $L/chain.log
