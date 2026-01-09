echo "Creating results directory structure..."
mkdir -p results/{robot_gripper_140N,package_drop_120N}
mkdir -p results/{tensile_test_330N,cable_tension_200N}

echo "Running Scenario 1: Robot Gripper Compression (140 N)..."
python material_pe_sensitivity.py \
    --mode compress \
    --force 140 \
    --out_dir results/robot_gripper_140N \
    --frames 50

echo "Running Scenario 2: Package Drop Test - 2 ft (120 N)..."
python material_pe_sensitivity.py \
    --mode compress \
    --force 120 \
    --out_dir results/package_drop_120N \
    --frames 50

echo "Running Scenario 3: Tensile Testing Machine (500 N)..."
python material_pe_sensitivity.py \
    --mode stretch \
    --force 330 \
    --out_dir results/tensile_test_330N \
    --frames 50

echo "Running Scenario 4: Cable/Wire Tension (200 N)..."
python material_pe_sensitivity.py \
    --mode stretch \
    --force 200 \
    --out_dir results/cable_tension_200N \
    --frames 50