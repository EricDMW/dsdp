#!/bin/bash
# View Latest Training Results
# This script helps you find and view the most recent training run results.

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

RUNS_DIR="/home/dongmingwang/project/DSDP/dsdp/wireless_comm/runs"

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}FINDING LATEST TRAINING RESULTS${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

# Find the latest run directory
LATEST_RUN=$(ls -td ${RUNS_DIR}/run_* 2>/dev/null | head -1)

if [ -z "$LATEST_RUN" ]; then
  echo -e "${YELLOW}No training runs found in ${RUNS_DIR}${NC}"
  exit 1
fi

RUN_NUMBER=$(basename "$LATEST_RUN" | grep -oP "run_\K\d+")

echo -e "${GREEN}Latest run: ${LATEST_RUN}${NC}"
echo -e "${GREEN}Run number: ${RUN_NUMBER}${NC}"
echo ""

# Check for lagrangian.csv (Phase 2 run)
LAGRANGIAN_CSV="${LATEST_RUN}/training_progress/lagrangian.csv"
if [ -f "$LAGRANGIAN_CSV" ]; then
  echo -e "${GREEN}✅ Phase 2 (Recording) run detected${NC}"
  echo ""
  
  # Check for shadow curve plots
  SHADOW_CURVES="${LATEST_RUN}/training_progress/lagrangian_shadow_curves.png"
  if [ -f "$SHADOW_CURVES" ]; then
    echo -e "${GREEN}Files available:${NC}"
    echo "  📊 CSV Data: ${LAGRANGIAN_CSV}"
    echo "  📈 Shadow Curves (PNG): ${SHADOW_CURVES}"
    echo "  📄 Shadow Curves (PDF): ${LATEST_RUN}/training_progress/lagrangian_shadow_curves.pdf"
    echo ""
    
    # Show first 10 lines of CSV
    echo -e "${BLUE}Preview of lagrangian.csv (first 10 lines):${NC}"
    echo "================================================================"
    head -10 "$LAGRANGIAN_CSV"
    echo "================================================================"
    echo ""
    
    # Show statistics
    LINE_COUNT=$(wc -l < "$LAGRANGIAN_CSV")
    echo -e "${GREEN}Total data points: $((LINE_COUNT - 1))${NC}"
    echo ""
    
  else
    echo -e "${YELLOW}⚠️  Shadow curve plots not found${NC}"
    echo -e "${YELLOW}   Generating plots now...${NC}"
    echo ""
    
    # Generate plots
    conda activate marl_env 2>/dev/null || source $(conda info --base)/etc/profile.d/conda.sh && conda activate marl_env
    python -m dsdp.wireless_comm.plot_lagrangian_shadow_curves --run-dir "$LATEST_RUN"
    
    echo ""
    echo -e "${GREEN}✅ Plots generated!${NC}"
  fi
  
  echo -e "${BLUE}Commands to view results:${NC}"
  echo "  # View CSV"
  echo "  cat ${LAGRANGIAN_CSV}"
  echo ""
  echo "  # View plots (if on desktop)"
  echo "  xdg-open ${SHADOW_CURVES} 2>/dev/null || open ${SHADOW_CURVES}"
  echo ""
  echo "  # Navigate to directory"
  echo "  cd ${LATEST_RUN}/training_progress"
  echo ""
  
elif [ -f "${LATEST_RUN}/info/final_lagrangian.json" ]; then
  echo -e "${GREEN}✅ Phase 1 (Reference) run detected${NC}"
  echo ""
  echo -e "${GREEN}Files available:${NC}"
  echo "  📋 Final Lagrangian: ${LATEST_RUN}/info/final_lagrangian.json"
  echo "  🤖 Models: ${LATEST_RUN}/model/agent_*_policy.pt"
  echo ""
  
  # Show final Lagrangian values
  if command -v jq &> /dev/null; then
    echo -e "${BLUE}Final Lagrangian Multipliers:${NC}"
    jq '.final_lagrangian_multipliers' "${LATEST_RUN}/info/final_lagrangian.json"
    echo ""
  else
    echo -e "${BLUE}Final Lagrangian data:${NC}"
    cat "${LATEST_RUN}/info/final_lagrangian.json"
    echo ""
  fi
  
  echo -e "${YELLOW}This is a reference run. Use it for Phase 2:${NC}"
  echo "  python -m dsdp.wireless_comm.example_loss_record_training \\"
  echo "    --phase 2 \\"
  echo "    --reference-run-dir ${LATEST_RUN} \\"
  echo "    --total-timesteps 50000"
  echo ""
  
else
  echo -e "${YELLOW}Standard training run (not Phase 1 or 2)${NC}"
  echo ""
  
  # Check for standard training log
  TRAINING_LOG=$(ls -t ${LATEST_RUN}/training_progress/training_log_*.csv 2>/dev/null | head -1)
  if [ -f "$TRAINING_LOG" ]; then
    echo -e "${GREEN}Training log found:${NC}"
    echo "  ${TRAINING_LOG}"
    echo ""
    head -5 "$TRAINING_LOG"
  fi
fi

echo -e "${BLUE}================================================================${NC}"

