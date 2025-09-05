#!/bin/bash

output_dirs=("z_valtest_real")

for output_dir in "${output_dirs[@]}"; do

    while true; do
        python validate_results.py --output_dir "$output_dir" \
                        
        if [ $? -eq 0 ]; then
            echo "✅ Run for $output_dir successful!"
            break
        else
            echo "⚠️ Run for $output_dir failed. Retrying in 5 seconds"
            sleep 5
        fi
    done

    while true; do
        python share_results.py --output_dir "$output_dir" \
                        
        if [ $? -eq 0 ]; then
            echo "✅ Sharing for $output_dir successful!"
            break
        else
            echo "⚠️ Sharing for $output_dir failed. Retrying in 5 seconds"
            sleep 5
        fi
    done
done