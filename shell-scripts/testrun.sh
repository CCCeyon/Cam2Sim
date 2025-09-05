#!/bin/bash

output_dirs=("longtest_fortuna_sim")

for output_dir in "${output_dirs[@]}"; do
    for ((i=0; i<=50; i+=10)); do  # increment changed to 10
        guidance_list=(3.0 3.5 4.0 4.5 5.0)
        for guidance in "${guidance_list[@]}"; do
            for seed_flag in true; do
                #for rotate_flag in true false; do  # new rotate flag loop
                for rotate_flag in false; do  # new rotate flag loop
                    
                    echo "=== Starte Run für $output_dir mit i=$i, guidance=$guidance, seed=$seed_flag, rotate=$rotate_flag ==="

                    # Build seed argument
                    seed_arg=""
                    seed_suffix="noseed"
                    if [ "$seed_flag" = true ]; then
                        seed_arg="--set_seed"
                        seed_suffix="seed"
                    fi

                    # Build rotate argument
                    rotate_arg=""
                    rotate_suffix="norotate"
                    if [ "$rotate_flag" = true ]; then
                        rotate_arg="--rotate"
                        rotate_suffix="rotate"
                    fi

                    target_folder="output/${output_dir}/output-${i}_g${guidance}_${seed_suffix}_${rotate_suffix}"

                    # ✅ Skip if already exists
                    if [ -d "$target_folder" ]; then
                        echo "⏭️  Skipping i=$i guidance=$guidance seed=$seed_flag rotate=$rotate_flag (already exists: $target_folder)"
                        continue
                    fi

                    while true; do
                        python redo_generation.py --split "$i" --output_dir "$output_dir" \
                            $seed_arg $rotate_arg --guidance "$guidance"
                        
                        #  --max_images 250
                        
                        if [ $? -eq 0 ]; then
                            echo "✅ Run for $output_dir i=$i guidance=$guidance seed=$seed_flag rotate=$rotate_flag successful!"
                            break
                        else
                            echo "⚠️ Run for $output_dir i=$i guidance=$guidance seed=$seed_flag rotate=$rotate_flag failed. Retrying in 5 seconds"
                            sleep 5
                        fi
                    done

                    while true; do
                        python export_results.py --name "$output_dir"
                        
                        if [ $? -eq 0 ]; then
                            echo "✅ Export for $output_dir i=$i guidance=$guidance seed=$seed_flag rotate=$rotate_flag successful!"
                            
                            if [ -f "exports/${output_dir}.mp4" ]; then
                                mv "exports/${output_dir}.mp4" "exports/${output_dir}-${i}_g${guidance}_${seed_suffix}_${rotate_suffix}.mp4"
                                echo "🎬 Renamed MP4 to ${output_dir}-${i}_g${guidance}_${seed_suffix}_${rotate_suffix}.mp4"
                            else
                                echo "⚠️ MP4 file ${output_dir}.mp4 not found after export!"
                            fi
                            
                            break
                        else
                            echo "⚠️ Export for $output_dir i=$i guidance=$guidance seed=$seed_flag rotate=$rotate_flag failed. Retrying in 5 seconds"
                            sleep 5
                        fi
                    done

                    # Rename folder to include i, guidance, seed, and rotate info
                    if [ -d "output/${output_dir}/output" ]; then
                        rm -rf "output/${output_dir}/output-${i}_g${guidance}_${seed_suffix}_${rotate_suffix}"
                        mv "output/${output_dir}/output" "output/${output_dir}/output-${i}_g${guidance}_${seed_suffix}_${rotate_suffix}"
                        echo "📁 Renamed to 'output/${output_dir}/output-${i}_g${guidance}_${seed_suffix}_${rotate_suffix}'"
                    else
                        echo "⚠️ No 'output/${output_dir}/output' folder found for i=$i guidance=$guidance seed=$seed_flag rotate=$rotate_flag"
                    fi

                done
            done
        done
    done
done

for output_dir in "${output_dirs[@]}"; do
    python validation.validate_vehicles --output_dir "$output_dir"
done