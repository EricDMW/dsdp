#!/bin/bash
# Two-Phase Training Script
# This script runs both phases of training with loss recording sequentially.

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default parameters
TOTAL_TIMESTEPS=80000
GRID_SIZE=5
ETA_MU=2.0
RHS=2.0
SEED=42  # Use same seed for both phases to ensure convergence

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --total-timesteps)
      TOTAL_TIMESTEPS="$2"
      shift 2
      ;;
    --grid-size)
      GRID_SIZE="$2"
      shift 2
      ;;
    --eta-mu)
      ETA_MU="$2"
      shift 2
      ;;
    --rhs)
      RHS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --total-timesteps NUM    Total timesteps for training (default: 50000)"
      echo "  --grid-size NUM          Grid size (default: 3)"
      echo "  --eta-mu NUM             Constraint weight (default: 2.0)"
      echo "  --rhs NUM                Constraint RHS (default: 2.0)"
      echo "  --seed NUM               Random seed for BOTH phases (default: 42)"
      echo "  --help                   Show this help message"
      echo ""
      echo "Example:"
      echo "  $0 --total-timesteps 100000 --grid-size 5"
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}TWO-PHASE TRAINING WITH LOSS RECORDING${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Total Timesteps: $TOTAL_TIMESTEPS"
echo "  Grid Size: $GRID_SIZE"
echo "  Eta Mu: $ETA_MU"
echo "  RHS: $RHS"
echo "  Seed (both phases): $SEED"
echo ""
echo -e "${YELLOW}⚠️  IMPORTANT: Using SAME seed for both phases ensures convergence!${NC}"
echo -e "${YELLOW}   Phase 1 and Phase 2 will use seed=$SEED${NC}"
echo -e "${YELLOW}   Expected: All errors → 0 (policies converge to reference)${NC}"
echo ""

# Activate conda environment
echo -e "${YELLOW}Activating marl_env...${NC}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate marl_env

# Verify setup
echo -e "${YELLOW}Verifying setup...${NC}"
python -m dsdp.wireless_comm.test_loss_record_setup
if [ $? -ne 0 ]; then
  echo -e "${RED}Setup verification failed!${NC}"
  exit 1
fi

echo ""
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}PHASE 1: REFERENCE TRAINING${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

# Run Phase 1
python -m dsdp.wireless_comm.example_loss_record_training \
  --phase 1 \
  --total-timesteps $TOTAL_TIMESTEPS \
  --grid-size $GRID_SIZE \
  --eta-mu $ETA_MU \
  --rhs $RHS \
  --seed $SEED \
  | tee phase1.log

# Extract run number from Phase 1 output
RUN_NUMBER=$(grep -oP "Run number: \K\d+" phase1.log | tail -1)

if [ -z "$RUN_NUMBER" ]; then
  echo -e "${RED}Failed to extract run number from Phase 1${NC}"
  exit 1
fi

echo ""
echo -e "${GREEN}Phase 1 completed successfully!${NC}"
echo -e "${GREEN}Run number: $RUN_NUMBER${NC}"
echo ""

# Construct reference directory path
REFERENCE_DIR="/home/dongmingwang/project/DSDP/dsdp/wireless_comm/runs/run_${RUN_NUMBER}"

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}PHASE 2: TRAINING WITH LOSS RECORDING${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""
echo -e "${GREEN}Reference directory: $REFERENCE_DIR${NC}"
echo ""

# Verify reference files exist
echo -e "${YELLOW}Verifying reference files...${NC}"
if [ ! -f "${REFERENCE_DIR}/info/final_lagrangian.json" ]; then
  echo -e "${RED}❌ Error: final_lagrangian.json not found!${NC}"
  exit 1
fi

# Count policy files
POLICY_COUNT=$(ls -1 ${REFERENCE_DIR}/model/agent_*_policy.pt 2>/dev/null | wc -l)
if [ "$POLICY_COUNT" -eq 0 ]; then
  echo -e "${RED}❌ Error: No policy files found!${NC}"
  exit 1
fi

echo -e "${GREEN}✅ Found final_lagrangian.json${NC}"
echo -e "${GREEN}✅ Found ${POLICY_COUNT} policy files (.pt)${NC}"
echo -e "${YELLOW}   Using seed=$SEED (SAME as Phase 1) for convergence${NC}"
echo ""

# Wait a moment before starting Phase 2
sleep 2

# Get timestamp before Phase 2
BEFORE_PHASE2=$(date +%s)

# Run Phase 2
python -m dsdp.wireless_comm.example_loss_record_training \
  --phase 2 \
  --reference-run-dir "$REFERENCE_DIR" \
  --total-timesteps $TOTAL_TIMESTEPS \
  --grid-size $GRID_SIZE \
  --eta-mu $ETA_MU \
  --rhs $RHS \
  --seed $SEED \
  | tee phase2.log

# Find the run directory created during Phase 2
# It should be the newest run directory created after Phase 1
RUNS_BASE_DIR="/home/dongmingwang/project/DSDP/dsdp/wireless_comm/runs"
RUN_NUMBER_PHASE2=""

# Find runs created after Phase 2 started
for run_dir in $(ls -td ${RUNS_BASE_DIR}/run_* 2>/dev/null); do
  run_time=$(stat -c %Y "$run_dir" 2>/dev/null || stat -f %m "$run_dir" 2>/dev/null)
  if [ $run_time -gt $BEFORE_PHASE2 ]; then
    # This run was created during Phase 2
    RUN_NUMBER_PHASE2=$(basename "$run_dir" | grep -oP "run_\K\d+")
    break
  fi
done

# If we couldn't find it by timestamp, try extracting from log
if [ -z "$RUN_NUMBER_PHASE2" ]; then
  RUN_NUMBER_PHASE2=$(grep -oP "Run number: \K\d+" phase2.log 2>/dev/null | tail -1)
fi

# If still not found, get the run number right after Phase 1
if [ -z "$RUN_NUMBER_PHASE2" ]; then
  RUN_NUMBER_PHASE2=$((RUN_NUMBER + 1))
fi

echo ""
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}TRAINING COMPLETE!${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""
echo -e "${GREEN}Phase 1 (Reference):${NC}"
echo "  Run number: $RUN_NUMBER"
echo "  Directory: $REFERENCE_DIR"
echo ""

PHASE2_DIR="${RUNS_BASE_DIR}/run_${RUN_NUMBER_PHASE2}"
LAGRANGIAN_CSV="${PHASE2_DIR}/training_progress/lagrangian.csv"

echo -e "${GREEN}Phase 2 (Recording):${NC}"
echo "  Run number: $RUN_NUMBER_PHASE2"
echo "  Directory: $PHASE2_DIR"
echo ""

# Check if lagrangian.csv exists to verify Phase 2 ran correctly
if [ -f "$LAGRANGIAN_CSV" ]; then
  echo -e "${GREEN}✅ Phase 2 completed successfully!${NC}"
  echo ""
  echo -e "${GREEN}Output files:${NC}"
  echo "  Lagrangian metrics: runs/run_${RUN_NUMBER_PHASE2}/training_progress/lagrangian.csv"
  echo "  Shadow curves: runs/run_${RUN_NUMBER_PHASE2}/training_progress/lagrangian_shadow_curves.png"
  echo "  Publication PDF: runs/run_${RUN_NUMBER_PHASE2}/training_progress/lagrangian_shadow_curves.pdf"
  echo ""
  
  # Show preview of results
  LINE_COUNT=$(wc -l < "$LAGRANGIAN_CSV" 2>/dev/null)
  if [ ! -z "$LINE_COUNT" ]; then
    echo -e "${GREEN}Data points recorded: $((LINE_COUNT - 1))${NC}"
  fi
  
  echo ""
  echo -e "${GREEN}To view results:${NC}"
  echo "  cd ${PHASE2_DIR}/training_progress"
  echo "  cat lagrangian.csv"
  echo ""
  echo "  # Or use helper script:"
  echo "  ./view_latest_results.sh"
else
  echo -e "${YELLOW}⚠️  Warning: lagrangian.csv not found${NC}"
  echo -e "${YELLOW}   Phase 2 may have run as standard training${NC}"
  echo ""
  echo "  Check: ${PHASE2_DIR}/training_progress/"
  echo "  For standard training logs instead of lagrangian metrics"
fi
echo ""

# Final summary
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}SUMMARY${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""
echo "Phase 1 (Reference run): run_${RUN_NUMBER}"
echo "Phase 2 (Recording run): run_${RUN_NUMBER_PHASE2}"
echo ""

if [ -f "$LAGRANGIAN_CSV" ]; then
  echo -e "${GREEN}✅ SUCCESS: Phase 2 loss recording completed${NC}"
else
  echo -e "${YELLOW}⚠️  WARNING: Phase 2 may not have recorded losses properly${NC}"
  echo ""
  echo "This can happen if:"
  echo "  1. Training was interrupted"
  echo "  2. Wrong trainer class was used"
  echo ""
  echo "To retry Phase 2 manually:"
  echo "  conda activate marl_env"
  echo "  python -m dsdp.wireless_comm.example_loss_record_training \\"
  echo "    --phase 2 \\"
  echo "    --reference-run-dir ${REFERENCE_DIR} \\"
  echo "    --total-timesteps ${TOTAL_TIMESTEPS} \\"
  echo "    --seed ${SEED}"
fi

echo ""
echo -e "${BLUE}================================================================${NC}"

# Cleanup log files
rm -f phase1.log phase2.log

echo -e "${GREEN}✅ Script completed!${NC}"

